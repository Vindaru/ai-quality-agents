"""A tiny in-process referral service: the system under test for generated tests.

It implements three of the golden stories (SMS opt-out, phone validation, partner status
updates) behind an HTTP-like ``request`` method, so generated tests read like API tests but
need no server, network or framework. ``BUGS`` lists deliberate defects; a service built with
one of them is a *mutant*. A good test suite passes on the correct service and fails on every
mutant (it "kills" it). Mutants a suite lets through show exactly which behaviour it never
checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

API_CONTRACT = """\
Referral service API (in-process; tests receive a `client` fixture).
  client.get(path) / client.post(path, json=..., headers=...) / client.put(path, json=...)
  Each returns a response with `.status` (int) and `.json` (dict).

Seed data: member M-1 (phone +15550100101, sms_opt_out false); referral R-1 (status SENT);
partner tokens: "tok-valid" (valid), "tok-expired" (expired).

POST /members/{id}/replies      {"text": str}
    "STOP" in any letter case, with surrounding spaces, opts the member out of SMS:
    200 {"sms_opt_out": true} and one consent_audit row {"member_id", "event": "opt_out",
    "timestamp"}. Other text: 200 {"sms_opt_out": <unchanged>}.
GET  /members/{id}              200 {"member_id", "phone", "sms_opt_out"} | 404
PUT  /members/{id}/phone        {"phone": str}
    Stored in E.164 (+1 and 10 digits). Fewer than 10 digits: 422 {"error": "phone_too_short"}.
    Letters or other invalid characters: 422 {"error": "phone_invalid"}. Success: 200 {"phone"}.
POST /outreach/sms              {"member_id": str, "message": str}
    Opted-out member: 409 {"error": "opted_out"}. Otherwise 202 {"to": <E.164 phone>}.
POST /referrals/{id}/status     {"status": str}, headers {"Authorization": <token>}
    Missing or expired token: 401. Unknown referral: 404. Otherwise 200 {"status"} and one
    referral_audit row {"referral_id", "status"}.
GET  /referrals/{id}            200 {"referral_id", "status"} | 404
GET  /audit/{table}             200 {"rows": [...]}   tables: consent_audit, referral_audit
"""

BUGS: dict[str, str] = {
    "stop_case_sensitive": "only upper-case STOP opts out",
    "no_consent_audit": "opt-out is not written to consent_audit",
    "sms_ignores_opt_out": "SMS is sent to opted-out members",
    "short_phone_accepted": "9-digit phone numbers are accepted",
    "wrong_invalid_code": "invalid phone numbers return 400 instead of 422",
    "no_e164": "phones are stored as typed, not normalised to E.164",
    "unknown_referral_200": "status update for an unknown referral returns 200",
    "expired_token_accepted": "expired partner tokens are accepted",
    "no_referral_audit": "status changes are not written to referral_audit",
}


@dataclass
class Response:
    status: int
    json: dict[str, Any] = field(default_factory=dict)


class ReferralService:
    def __init__(self, bug: str | None = None) -> None:
        if bug is not None and bug not in BUGS:
            raise ValueError(f"unknown bug {bug!r}")
        self.bug = bug
        self.members = {"M-1": {"member_id": "M-1", "phone": "+15550100101", "sms_opt_out": False}}
        self.referrals = {"R-1": {"referral_id": "R-1", "status": "SENT"}}
        self.tokens = {"tok-valid": True, "tok-expired": False}
        self.audit: dict[str, list[dict[str, Any]]] = {"consent_audit": [], "referral_audit": []}
        self.clock = 0

    # --- routing -----------------------------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        body, headers = body or {}, headers or {}
        routes = [
            ("POST", r"/members/([^/]+)/replies", self._reply),
            ("GET", r"/members/([^/]+)", self._get_member),
            ("PUT", r"/members/([^/]+)/phone", self._set_phone),
            ("POST", r"/outreach/sms", self._send_sms),
            ("POST", r"/referrals/([^/]+)/status", self._set_status),
            ("GET", r"/referrals/([^/]+)", self._get_referral),
            ("GET", r"/audit/([^/]+)", self._get_audit),
        ]
        for verb, pattern, handler in routes:
            match = re.fullmatch(pattern, path)
            if verb == method.upper() and match:
                return handler(*match.groups(), body=body, headers=headers)
        return Response(404, {"error": "no_route"})

    # --- handlers ----------------------------------------------------------------------------

    def _reply(self, member_id: str, *, body: dict, headers: dict) -> Response:
        member = self.members.get(member_id)
        if member is None:
            return Response(404, {"error": "unknown_member"})
        text = str(body.get("text", "")).strip()
        is_stop = text == "STOP" if self.bug == "stop_case_sensitive" else text.upper() == "STOP"
        if is_stop:
            member["sms_opt_out"] = True
            if self.bug != "no_consent_audit":
                self.clock += 1
                self.audit["consent_audit"].append(
                    {"member_id": member_id, "event": "opt_out", "timestamp": self.clock}
                )
        return Response(200, {"sms_opt_out": member["sms_opt_out"]})

    def _get_member(self, member_id: str, *, body: dict, headers: dict) -> Response:
        member = self.members.get(member_id)
        return Response(200, dict(member)) if member else Response(404, {"error": "unknown"})

    def _normalise(self, phone: str) -> tuple[str | None, str | None]:
        if re.search(r"[^\d\s()+.\-]", phone):
            return None, "phone_invalid"
        digits = re.sub(r"\D", "", phone)
        digits = digits[1:] if len(digits) == 11 and digits.startswith("1") else digits
        min_digits = 9 if self.bug == "short_phone_accepted" else 10
        if len(digits) < min_digits:
            return None, "phone_too_short"
        if len(digits) > 10:
            return None, "phone_invalid"
        return (phone if self.bug == "no_e164" else f"+1{digits.rjust(10, '0')}"), None

    def _set_phone(self, member_id: str, *, body: dict, headers: dict) -> Response:
        member = self.members.get(member_id)
        if member is None:
            return Response(404, {"error": "unknown_member"})
        phone, error = self._normalise(str(body.get("phone", "")))
        if error:
            return Response(400 if self.bug == "wrong_invalid_code" else 422, {"error": error})
        member["phone"] = phone
        return Response(200, {"phone": phone})

    def _send_sms(self, *, body: dict, headers: dict) -> Response:
        member = self.members.get(str(body.get("member_id")))
        if member is None:
            return Response(404, {"error": "unknown_member"})
        if member["sms_opt_out"] and self.bug != "sms_ignores_opt_out":
            return Response(409, {"error": "opted_out"})
        return Response(202, {"to": member["phone"]})

    def _set_status(self, referral_id: str, *, body: dict, headers: dict) -> Response:
        token = headers.get("Authorization")
        valid = self.tokens.get(token)
        if not valid and not (self.bug == "expired_token_accepted" and token in self.tokens):
            return Response(401, {"error": "unauthorized"})
        referral = self.referrals.get(referral_id)
        if referral is None:
            if self.bug == "unknown_referral_200":
                return Response(200, {"status": body.get("status")})
            return Response(404, {"error": "unknown_referral"})
        referral["status"] = str(body.get("status"))
        if self.bug != "no_referral_audit":
            self.audit["referral_audit"].append(
                {"referral_id": referral_id, "status": referral["status"]}
            )
        return Response(200, {"status": referral["status"]})

    def _get_referral(self, referral_id: str, *, body: dict, headers: dict) -> Response:
        referral = self.referrals.get(referral_id)
        return Response(200, dict(referral)) if referral else Response(404, {"error": "unknown"})

    def _get_audit(self, table: str, *, body: dict, headers: dict) -> Response:
        if table not in self.audit:
            return Response(404, {"error": "unknown_table"})
        return Response(200, {"rows": [dict(r) for r in self.audit[table]]})


FAULTS: dict[str, dict[str, Any]] = {
    "gateway_down": {
        "category": "environment",
        "logs": ["ingress: upstream referral-service not ready", "health check failing for 4m"],
    },
    "seed_missing": {
        "category": "test_data",
        "logs": ["seed job finished: 0 members loaded (expected 1)"],
    },
}


class ServiceClient:
    """What a generated test sees as its ``client`` fixture. ``fault`` simulates a broken test
    environment rather than a broken service: the gateway is down, or seed data never loaded."""

    def __init__(self, service: ReferralService, fault: str | None = None) -> None:
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}")
        self._service = service
        self._fault = fault

    def _call(self, method: str, path: str, body: Any, headers: Any) -> Response:
        if self._fault == "gateway_down":
            raise ConnectionRefusedError("[Errno 111] Connection refused: referral-service.test")
        touches_m1 = "M-1" in path or (body or {}).get("member_id") == "M-1"
        if self._fault == "seed_missing" and touches_m1:
            raise LookupError("seed data missing: member M-1 was not loaded")
        return self._service.request(method, path, body, headers)

    def get(self, path: str, headers: dict[str, str] | None = None) -> Response:
        return self._call("GET", path, None, headers)

    def post(
        self, path: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Response:
        return self._call("POST", path, json, headers)

    def put(
        self, path: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Response:
        return self._call("PUT", path, json, headers)

import inspect
import asyncio
import json
import logging
import os
import time
from typing import Any
from utils.core.log import setup_logging
from utils.core.warnings_config import configure_warning_filters
from flask import Flask, request, jsonify
from utils.core.slack import SlackActivityLogger, SlackActivityMeta
from utils.llm.gcp_credentials import ensure_gcp_credentials_from_vault

configure_warning_filters()
ensure_gcp_credentials_from_vault()

from tools.chat.chat import chat_main
from tools.tech_rfp.tech_rfp import tech_rfp_analysis_main
from tools.tech_rfp.rfp_summarizer import rfp_summary_main
from tools.contract.contract_review import contract_review_main
from tools.tech_rfp.tech_rfp_generate_report import tech_report_main
from tools.tech_rfp.tech_rfp_generate_evaluation import generate_eval_main
from tools.tech_rfp.tech_rfp_evaluation_criteria_extractor import (
    eval_analysis_main,
    evaluation_fetch_main,
)
from tools.comm_rfp.comm_rfp import comm_rfp_main

app = Flask(__name__)
setup_logging()
logger = logging.getLogger("TruBuildBE")

"""
API for TruBuild Backend

pip install flask
"""


_LAST = {"status": None, "t": 0.0}
GET_INFO_EVERY_SEC = 300


def _should_log_get(current_status: str) -> bool:
    now = time.monotonic()
    if _LAST["status"] != current_status or now - _LAST["t"] >= GET_INFO_EVERY_SEC:
        _LAST["status"] = current_status
        _LAST["t"] = now
        return True
    return False


def handle(tool_func=None, *args, **kwargs):
    """
    Universal wrapper for all endpoint tools.

    - Expects the caller (each route) to pass ALL parameters required
      by the tool function through *args / **kwargs.
    - Automatically builds the standard response envelope required by
      Trubuild's API spec.
    - Does not enforce a fixed whitelist of status values - it leaves
      whatever the tool returns, or falls back to "done"/"error".
    """
    req_json = kwargs.pop("request_body", {})
    remote_ip = request.remote_addr
    user_id = req_json.get("userId", "")
    tool_name = tool_func.__name__ if tool_func else "unknown_tool"
    package_id = req_json.get("packageId", "unknown")
    user_name = req_json.get("userName", "")
    method = request.method

    context = {
        "tool_name": tool_name,
        "ip_address": remote_ip,
        "package_id": package_id,
        "request_type": method,
        "user_name": user_name,
    }
    logger = logging.LoggerAdapter(logging.getLogger("TruBuildBE"), context)

    if method == "POST":
        logger.info("Process started")
    elif method not in ("GET",):
        logger.info("Invoke via %s: %s (user=%s)", method, tool_name, user_id)

    response = {
        "userId": user_id,  # always echo back
        "status": "",  # will be set below
        "error": "",
        "tokens": 0,  # default 0 when unknown
        "toolData": {},  # populated on success
    }

    call_kwargs = dict(kwargs)
    sig = inspect.signature(tool_func) if tool_func else None
    if sig:
        if "remote_ip" in sig.parameters:
            call_kwargs["remote_ip"] = remote_ip
        if "request_method" in sig.parameters:
            call_kwargs["request_method"] = method

    try:
        # invoke the actual tool function
        if asyncio.iscoroutinefunction(tool_func):
            result = asyncio.run(tool_func(*args, **call_kwargs))
        else:
            result = tool_func(*args, **call_kwargs)

    except Exception as exc:
        logger.exception(f"{tool_name} crashed")
        response["status"] = "error"
        response["error"] = str(exc)
        return jsonify(response), 500

    # normalise tool output
    #
    # We expect each tool to return:
    #   {
    #       "status": "done" | "in progress" | "error",
    #       "tokens": <int>,          # optional
    #       ... <arbitrary payload>   # everything else = toolData
    #   }
    # If the tool returns plain data (not a dict), we still wrap it.
    if isinstance(result, dict):
        # pull optional keys out; the rest becomes toolData
        response["tokens"] = result.pop("tokens", 0)
        response["status"] = result.pop("status", "done")

        if "error" in result:
            # tool reported its own error
            response["error"] = result.pop("error")
        else:
            response["toolData"] = result
    else:
        # non-dict return -> treat as successful payload
        response["status"] = "done" if result else "error"
        response["toolData"] = result

    if method == "GET":
        current_status = response.get("status") or (
            "error" if response.get("error") else "done"
        )
        if _should_log_get(current_status):
            logger.info("Status check: %s", current_status)

    # done
    return jsonify(response), 200


def bad_request(msg: str, user_id: str = ""):
    envelope = {
        "userId": user_id,
        "status": "error",
        "error": msg,
        "tokens": 0,
        "toolData": {},
    }
    return jsonify(envelope), 400


def get_payload() -> dict:
    """
    Return the request payload as a dict.
    - POST   – accept plaintext JSON.
    - GET    – parse flat query params and group known toolData fields.
    """
    if request.method == "GET":
        args = request.args.to_dict(flat=True) if request.args else {}

        # all valid toolData keys from all endpoints
        tool_keys = {
            "prompt",
            "conversationId",
            "documents",
            "internet",
            "contractType",
            "paymentOption",
            "governedLaw",
            "analysisType",
        }

        # move recognized toolData keys into a nested dict
        tool_data = {k: args.pop(k) for k in list(args) if k in tool_keys}
        if tool_data:
            args["toolData"] = tool_data

        return args

    # POST requests - accept plaintext JSON
    return request.get_json(force=True, silent=True) or {}


def ping_status_tool(
    package_id: str | None = None,
    request_method: str | None = None,
    remote_ip: str | None = None,
    user_name: str | None = None,
) -> dict:
    """
    Healthcheck tool.
    - Returns {"status": "pong"} (wrapped by handle()).
    - Logs an INFO line into activity.log.
    - Sends a Slack activity message.
    """
    from utils.core.log import pid_tool_logger, get_logger, set_logger

    base_logger = pid_tool_logger(package_id=package_id, tool_name="ping")
    set_logger(
        base_logger,
        tool_name="ping",
        tool_base="ping",
        package_id=package_id or "unknown",
        ip_address=remote_ip or "no_ip",
        request_type=request_method or "N/A",
    )
    logger = get_logger()
    logger.info("Ping received; replying with pong")

    # Slack activity (never fail the endpoint if Slack is misconfigured)
    try:
        s = SlackActivityLogger(
            SlackActivityMeta(package_id=package_id, tool="PING", user=user_name)
        )
        thread_ts = s.start()
        s.sub(f"PONG")
        s.done("pong")
    except Exception as slack_exc:
        # Log Slack issues to activity.log but do not break /ping
        logger.error(f"Slack logging skipped or failed: {slack_exc}")

    return {"status": "pong"}


@app.route("/ping", methods=["GET", "POST"])
def PING():
    data = get_payload()
    return handle(
        tool_func=ping_status_tool,
        request_body=data,
        package_id=data.get("packageId"),
        user_name=data.get("userName") or data.get("user"),
    )


# chat has only POST
@app.route("/chat", methods=["POST"])
def CHAT():
    """
    TruBuild Chat endpoint. No GET supported.
    """
    data = get_payload()
    td = data.get("toolData", {})
    prompt = td.get("prompt")
    package_id = data.get("packageId")
    company_id = data.get("companyId")
    documents = td.get("documents", [])
    country_code = data.get("countryCode", "USA")
    conversation = td.get("conversationId", "default")
    internet_flag = bool(td.get("internet", False))

    if not prompt or not (package_id or company_id):
        return bad_request(
            "prompt is required, and either packageId/pkgID or companyId must be provided",
            data.get("userId", ""),
        )

    # Let handle() wrap the call and return the standard envelope
    response = handle(
        tool_func=chat_main,
        request_body=data,
        package_id=package_id,
        user_name=data.get("userName", ""),
        conversation_id=conversation,
        prompt=prompt,
        documents=documents,
        internet=internet_flag,
        country_code=country_code,
        company_id=company_id,
        request_method=request.method,
    )

    return response


@app.route("/review", methods=["GET", "POST"])
def CONTRACT_REVIEW():
    """
    Contract analyzer and reviewer endpoint
    - GET - fetch finished JSONs if present
    - POST - kick off analysis + review in the background
    """

    data = get_payload()
    package_id = data.get("packageId")
    company_id = data.get("companyId")
    country_code = data.get("countryCode", "USA")
    td = data.get("toolData", {})
    compute_flag = bool(data.get("computeReanalysis", True))
    user_id = data.get("userId")
    package_name = data.get("packageName")

    if not package_id:
        return bad_request("packageId is required", data.get("userId", ""))

    response = handle(
        tool_func=contract_review_main,
        request_body=data,
        package_id=package_id,
        user_name=data.get("userName", ""),
        contract_type=td.get("contractType", "NEC"),
        payment_option=td.get("paymentOption", "A"),
        governed_law=td.get("governedLaw", "USA"),
        country_code=country_code,
        request_method=request.method,
        user_type=data.get("userType", "client"),
        compute_reanalysis=compute_flag,
        company_id=company_id,
        user_id=user_id,
        package_name=package_name,
    )
    return response


@app.route("/tech-rfp", methods=["GET", "POST"])
def TECH_RFP():
    data = get_payload()
    tool_data = data.get("toolData", {}) or {}
    package_id = data.get("packageId")
    company_id = data.get("companyId")
    country_code = data.get("countryCode", "USA")
    compute_flag = bool(data.get("computeReanalysis", True))
    user_id = data.get("userId")
    package_name = data.get("packageName")
    evaluation_criteria = (
        tool_data.get("evaluation_criteria")
        or tool_data.get("evaluationCriteria")
        or data.get("evaluation_criteria")
        or data.get("evaluationCriteria")
    )
    if not package_id:
        return bad_request("packageId is required", data.get("userId", ""))

    raw_type = tool_data.get("analysisType") or data.get("analysisType")
    if not raw_type:
        return bad_request(
            "analysisType is required for all requests", data.get("userId", "")
        )

    analysis_type = raw_type.lower()
    get_dispatch = {
        "evaluation": (
            evaluation_fetch_main,
            {},
        ),  # merged fetch during GET for extract-eval and generate eval
        "summary": (rfp_summary_main, {"rfp_variant": "tech"}),
        "report": (tech_report_main, {}),
        "analysis": (tech_rfp_analysis_main, {}),
    }

    post_dispatch = {
        "summary": (rfp_summary_main, {"rfp_variant": "tech"}),
        "report": (tech_report_main, {}),
        "analysis": (tech_rfp_analysis_main, {}),
        "extract-eval": (eval_analysis_main, {}),  # POST only
        "generate-eval": (generate_eval_main, {}),  # POST only
    }

    dispatch = get_dispatch if request.method == "GET" else post_dispatch
    if analysis_type not in dispatch:
        allowed = ", ".join(dispatch.keys())
        return bad_request(
            f"Unsupported analysisType for {request.method}: {analysis_type}. Allowed: {allowed}",
            data.get("userId", ""),
        )

    tool_func, extra_kwargs = dispatch[analysis_type]

    response = handle(
        tool_func=tool_func,
        request_body=data,
        package_id=package_id,
        country_code=country_code,
        request_method=request.method,
        compute_reanalysis=compute_flag,
        company_id=company_id,
        user_name=data.get("userName", ""),
        user_id=user_id,
        package_name=package_name,
        evaluation_criteria=evaluation_criteria,
        **extra_kwargs,
    )

    return response


@app.route("/comm-rfp", methods=["GET", "POST"])
def COMM_RFP():
    data = get_payload()
    tool_data = data.get("toolData", {}) or {}
    package_id = data.get("packageId")
    company_id = data.get("companyId")
    country_code = data.get("countryCode", "USA")
    compute_flag = bool(data.get("computeReanalysis", True))
    user_id = data.get("userId")
    package_name = data.get("packageName")

    if not package_id:
        return bad_request("packageId is required", data.get("userId", ""))

    raw_type = tool_data.get("analysisType") or data.get("analysisType")
    if not raw_type:
        return bad_request(
            "analysisType is required for all requests", data.get("userId", "")
        )

    analysis_type = raw_type.lower()
    dispatch = {
        "extract": (comm_rfp_main, {"analysis_type": "extract"}),
        "compare": (comm_rfp_main, {"analysis_type": "compare"}),
    }
    if analysis_type not in dispatch:
        allowed = ", ".join(dispatch.keys())
        return bad_request(
            f"Unsupported analysisType for {request.method}: {analysis_type}. Allowed: {allowed}",
            data.get("userId", ""),
        )

    tool_func, extra_kwargs = dispatch[analysis_type]
    response = handle(
        tool_func=tool_func,
        request_body=data,
        package_id=package_id,
        country_code=country_code,
        request_method=request.method,
        compute_reanalysis=compute_flag,
        company_id=company_id,
        user_name=data.get("userName", ""),
        user_id=user_id,
        package_name=package_name,
        **extra_kwargs,
    )
    return response

# @app.route('/delete', methods=['GET', 'POST'])
# @app.route('/usr', methods=['GET', 'POST'])

if __name__ == "__main__":
    if os.path.exists("crt.pem") and os.path.exists("key.pem"):
        app.run(host="0.0.0.0", port=5000, ssl_context=("crt.pem", "key.pem"))
    else:
        app.run(host="0.0.0.0", port=5000)

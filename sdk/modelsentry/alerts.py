"""Email alert module for ModelSentry drift events.

Sends plain-text SMTP email when a DriftReport exceeds a severity threshold.
Uses stdlib smtplib only — no external dependencies.

Designed to be called from the storage layer via a registered callback so
that alerting fires wherever save_drift_report is called, not just from the
server.
"""
from __future__ import annotations

import logging
import math
import smtplib
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import Literal

from modelsentry.drift import DriftReport

log = logging.getLogger(__name__)

_SEVERITY_ORDER: dict[str, int] = {"warning": 0, "critical": 1}


@dataclass
class AlertConfig:
    """SMTP configuration and filtering rules for drift email alerts.

    Args:
        recipient_email: Address that receives the alert.
        smtp_host: SMTP server hostname. Default: smtp.gmail.com (STARTTLS).
        smtp_port: SMTP port. Default: 587 (STARTTLS). Use 465 for SSL.
        smtp_user: SMTP login username (usually the sender address).
        smtp_password: SMTP login password or app password.
        from_email: Sender address in the email header. Defaults to smtp_user.
        min_severity: Minimum severity that triggers an alert.
        model_id_filter: If set, only alert for these model IDs. None = all models.
    """

    recipient_email: str
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str | None = None
    min_severity: Literal["warning", "critical"] = "warning"
    model_id_filter: list[str] | None = None


def _severity_meets_threshold(severity: str, min_severity: str) -> bool:
    return _SEVERITY_ORDER.get(severity, -1) >= _SEVERITY_ORDER[min_severity]


def _build_subject(model_id: str, severity: str) -> str:
    return f"ModelSentry Alert: {model_id} — {severity} drift detected"


def _build_body(report: DriftReport, model_id: str) -> str:
    lines = [
        f"ModelSentry detected {report.overall_severity} drift in model: {model_id}",
        "",
        "Drifting features:",
    ]
    for name, result in sorted(report.feature_results.items()):
        if result.severity in ("warning", "critical"):
            psi = result.psi
            psi_str = f"{psi:.4f}" if psi is not None and math.isfinite(psi) else "n/a"
            ks_str = (
                f"{result.ks_p_value:.4f}" if result.ks_p_value is not None else "n/a"
            )
            lines.append(
                f"  {name}: severity={result.severity}, PSI={psi_str}, KS p={ks_str}"
            )
    if report.missing_in_current:
        lines += ["", f"Features missing in current: {', '.join(report.missing_in_current)}"]
    if report.missing_in_baseline:
        lines += ["", f"Features new in current: {', '.join(report.missing_in_baseline)}"]
    lines += ["", "—", "ModelSentry local dashboard: http://127.0.0.1:8080"]
    return "\n".join(lines)


def send_drift_alert(report: DriftReport, model_id: str, config: AlertConfig) -> bool:
    """Send a drift alert email via SMTP.

    Returns True if the email was sent successfully, False on any failure.
    Never raises — failures are logged as warnings.

    Args:
        report: The DriftReport to summarise in the email.
        model_id: Identifier of the monitored model.
        config: SMTP connection settings and filtering rules.
    """
    if not _severity_meets_threshold(report.overall_severity, config.min_severity):
        return False
    if config.model_id_filter is not None and model_id not in config.model_id_filter:
        return False

    from_addr = config.from_email or config.smtp_user
    msg = MIMEText(_build_body(report, model_id), "plain")
    msg["Subject"] = _build_subject(model_id, report.overall_severity)
    msg["From"] = from_addr
    msg["To"] = config.recipient_email

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as smtp:
            smtp.starttls()
            smtp.login(config.smtp_user, config.smtp_password)
            smtp.sendmail(from_addr, [config.recipient_email], msg.as_string())
        log.info("Drift alert sent for model=%s severity=%s", model_id, report.overall_severity)
        return True
    except Exception as exc:
        log.warning("Failed to send drift alert for model=%s: %s", model_id, exc)
        return False

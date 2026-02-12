import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
audit_logger = logging.getLogger("ecosmart-audit")

class AuditService:
    @staticmethod
    def log_access(user_id, session_id, action):
        audit_logger.info(f"AUDIT | {datetime.now().isoformat()} | User: {user_id} | Session: {session_id} | Action: {action}")

    @staticmethod
    def log_clinical_decision(session_id, decision, confidence):
        audit_logger.info(f"CLINICAL | {datetime.now().isoformat()} | Session: {session_id} | Decision: {decision} | Conf: {confidence}")

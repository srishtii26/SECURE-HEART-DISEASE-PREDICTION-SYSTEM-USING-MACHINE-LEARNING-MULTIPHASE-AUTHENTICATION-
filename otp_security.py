import secrets
import time
import hashlib

OTP_EXPIRY_SECONDS = 300       # 5 minutes
MAX_ATTEMPTS = 3
BLOCK_SECONDS = 600             # 10 minutes


def generate_otp():
    """Generate a cryptographically secure 6-digit OTP."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp):
    """Hash OTP before storing it."""
    return hashlib.sha256(otp.encode()).hexdigest()


def create_otp():
    """Create a new OTP record."""
    otp = generate_otp()

    return {
        "otp_hash": hash_otp(otp),
        "created_at": time.time(),
        "expires_at": time.time() + OTP_EXPIRY_SECONDS,
        "attempts": 0,
        "verified": False,
        "blocked_until": 0,
    }, otp


def verify_otp(record, entered_otp):
    """
    Verify OTP with expiry and attempt protection.

    Returns:
        success: bool
        message: str
    """

    if record is None:
        return False, "No OTP is currently active."

    now = time.time()

    # Temporary block
    if now < record.get("blocked_until", 0):
        remaining = int(record["blocked_until"] - now)
        return False, f"Too many failed attempts. Try again in {remaining} seconds."

    # Expiry check
    if now > record["expires_at"]:
        return False, "OTP has expired. Please request a new OTP."

    # Maximum attempts
    if record["attempts"] >= MAX_ATTEMPTS:
        record["blocked_until"] = now + BLOCK_SECONDS
        return False, "Maximum attempts reached. Temporarily blocked."

    # Count attempt
    record["attempts"] += 1

    # Verify hashed OTP
    if secrets.compare_digest(
        record["otp_hash"],
        hash_otp(entered_otp)
    ):
        record["verified"] = True
        return True, "OTP verified successfully."

    # Block immediately after third failure
    if record["attempts"] >= MAX_ATTEMPTS:
        record["blocked_until"] = now + BLOCK_SECONDS
        return False, "Maximum attempts reached. Temporarily blocked."

    remaining = MAX_ATTEMPTS - record["attempts"]

    return False, f"Invalid OTP. {remaining} attempt(s) remaining."


def is_expired(record):
    """Check whether the OTP has expired."""
    if record is None:
        return True

    return time.time() > record["expires_at"]
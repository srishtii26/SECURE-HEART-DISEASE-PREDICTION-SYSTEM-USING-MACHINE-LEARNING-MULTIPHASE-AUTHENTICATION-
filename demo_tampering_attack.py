import os
import shutil
import hashlib


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/heart_model.pkl"
BACKUP_PATH = "models/heart_model_tampered_backup.pkl"


# ==========================================
# SHA-256 HASH FUNCTION
# ==========================================

def calculate_hash(file_path):

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(4096),
            b""
        ):

            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


# ==========================================
# CHECK ORIGINAL MODEL
# ==========================================

print("\n==========================================")
print("MODEL TAMPERING ATTACK DEMONSTRATION")
print("==========================================\n")


if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found:")
    print(MODEL_PATH)
    exit()


original_hash = calculate_hash(
    MODEL_PATH
)


print("Original Model SHA-256:")
print(original_hash)


# ==========================================
# CREATE BACKUP
# ==========================================

shutil.copy2(
    MODEL_PATH,
    BACKUP_PATH
)


print("\nBackup created.")


# ==========================================
# SIMULATE ATTACK
# ==========================================

print("\nSimulating model tampering...")

with open(
    MODEL_PATH,
    "ab"
) as file:

    file.write(
        b"TAMPERED"
    )


# ==========================================
# CALCULATE NEW HASH
# ==========================================

tampered_hash = calculate_hash(
    MODEL_PATH
)


print("\nTampered Model SHA-256:")
print(tampered_hash)


# ==========================================
# COMPARE HASHES
# ==========================================

print("\n==========================================")

if original_hash == tampered_hash:

    print("SECURITY FAILURE")
    print("Hashes are identical.")
    print("Tampering was NOT detected.")

else:

    print("TAMPERING DETECTED")
    print("SHA-256 hash mismatch confirmed.")
    print("Prediction should be BLOCKED.")

print("==========================================")


# ==========================================
# RESTORE ORIGINAL MODEL
# ==========================================

shutil.copy2(
    BACKUP_PATH,
    MODEL_PATH
)

os.remove(
    BACKUP_PATH
)


restored_hash = calculate_hash(
    MODEL_PATH
)


print("\nOriginal model restored.")

print("\nRestored Model SHA-256:")
print(restored_hash)


if restored_hash == original_hash:

    print("\nRESTORATION SUCCESSFUL")
    print("The original model is intact.")

else:

    print("\nWARNING")
    print("Restored model hash does not match original.")
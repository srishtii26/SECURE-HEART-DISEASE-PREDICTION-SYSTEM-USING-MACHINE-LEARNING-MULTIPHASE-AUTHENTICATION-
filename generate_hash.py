import hashlib


MODEL_PATH = "models/heart_model.pkl"


def calculate_hash(file_path):

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(4096),
            b""
        ):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


model_hash = calculate_hash(
    MODEL_PATH
)

print("Trusted Model SHA-256 Hash:")
print(model_hash)
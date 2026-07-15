# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Signature verification for dynamically loaded Python files.

Each dynamically loaded ``.py`` must have a companion ``.py.sign`` RSA
signature, verified with the public key below before execution. The public key
can be committed safely (it cannot forge signatures); only the private key
holder can sign files. Sign with
``python akg/utils/signature_verify.py --sign --private-key KEY``.
"""
import argparse
import base64
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

_SIGN_SUFFIX = ".sign"


# Verification code: RSA-2048 public key used to verify .py.sign files.
# Safe to commit: knowing this key cannot help an attacker forge signatures.
# Override at runtime by setting AKG_VERIFY_PUBLIC_KEY to a PEM file path.
VERIFY_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAs0LkVRudpBQtezqe7wy/
2cokj0KT+jQeyS+plPWNsvRTYM/ZDE8gSnqSWhhL6okUmoOo7c+eGsvR4q+F0o5e
ibLKPgIZ3ZjfJfzPqcQEwFJh2mLySaqfFm1KYzTOHF6tKEDniiCsaX1czv4ePYnq
ge9s4mvndQiDiqRsHLumMJMFHDKQIsI00cmYBPeUb38Epu8zCgJOVpALBTJ2boTe
r5QnNAN3xkznAAhiVSJb7mQRLsPA7A1HUu0xaGrMY6TEPw8cb7Qk92M1CA+0A6On
23PM46NeVHiq4DdOu4GrhsiJlZB7Kuf7LXBF/YUPbUGXhQdLRJ8ytScPpvWOVbCG
6wIDAQAB
-----END PUBLIC KEY-----
"""


def _load_public_key():
    """Load the verification public key.

    Priority: AKG_VERIFY_PUBLIC_KEY env var (path to PEM file) > built-in
    default VERIFY_PUBLIC_KEY_PEM.
    """
    key_path = os.environ.get("AKG_VERIFY_PUBLIC_KEY")
    if key_path:
        with open(key_path, "rb") as f:
            return serialization.load_pem_public_key(f.read())
    return serialization.load_pem_public_key(VERIFY_PUBLIC_KEY_PEM)


_PUBLIC_KEY = _load_public_key()


def get_sign_path(py_path):
    """Return the companion ``.py.sign`` path for ``py_path``."""
    return py_path + _SIGN_SUFFIX


def verify_file_signature(py_path):
    """Return True if ``py_path`` has a valid ``.py.sign`` companion file."""
    sign_path = get_sign_path(py_path)
    if not os.path.isfile(py_path) or not os.path.isfile(sign_path):
        return False
    with open(py_path, "rb") as f:
        content = f.read()
    with open(sign_path, "rb") as f:
        signature = f.read()
    try:
        _PUBLIC_KEY.verify(signature, content, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        return False
    return True


def verify_file_signature_or_raise(py_path):
    """Verify ``py_path`` signature, raising RuntimeError on failure."""
    if not verify_file_signature(py_path):
        raise RuntimeError(
            "Signature verification failed for '{}'. A valid '{}' file signed "
            "by the trusted key is required.".format(py_path, get_sign_path(py_path)))


def sign_file(py_path, private_key):
    """Sign ``py_path`` and write the signature to ``py_path + '.sign'``."""
    with open(py_path, "rb") as f:
        content = f.read()
    signature = private_key.sign(content, padding.PKCS1v15(), hashes.SHA256())
    sign_path = get_sign_path(py_path)
    with open(sign_path, "wb") as f:
        f.write(signature)
    return sign_path


def sign_bytes(content, private_key):
    """Sign raw bytes and return a base64-encoded signature string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    signature = private_key.sign(content, padding.PKCS1v15(), hashes.SHA256())
    return base64.b64encode(signature).decode("ascii")


def _load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _generate_keypair(private_key_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    with open(private_key_path, "wb") as f:
        f.write(priv_pem)
    try:
        os.chmod(private_key_path, 0o600)
    except OSError:
        pass
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def main():
    parser = argparse.ArgumentParser(
        description="Generate .py.sign signature files for dynamically loaded akg modules.")
    parser.add_argument("--genkey", action="store_true",
                        help="Generate a new RSA key pair (prints public key PEM).")
    parser.add_argument("--sign", metavar="FILE",
                        help="Sign FILE, producing a companion .py.sign file.")
    parser.add_argument("--private-key", metavar="PATH",
                        default=os.environ.get("AKG_SIGN_PRIVATE_KEY"),
                        help="RSA private key PEM (defaults to AKG_SIGN_PRIVATE_KEY env var).")
    args = parser.parse_args()

    if args.genkey:
        if not args.private_key:
            parser.error("--private-key is required for --genkey")
        _, pub_pem = _generate_keypair(args.private_key)
        print("Private key saved to: {}".format(args.private_key))
        print("Public key PEM (paste into VERIFY_PUBLIC_KEY_PEM):")
        print(pub_pem.decode())
        return

    if args.sign:
        if not args.private_key:
            parser.error("--private-key is required for --sign")
        sign_path = sign_file(args.sign, _load_private_key(args.private_key))
        print("Signature written to: {}".format(sign_path))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

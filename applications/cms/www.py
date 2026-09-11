"""Publish plots to the CERN www area from CBE, via Kerberos and xrdcp.

Usage, from the repo root on CBE:

    python -m applications.cms.www <subdir> <files...>

or from another script:

    from applications.cms.www import publish
    publish(["plot.png", "plot.pdf"], "nefqvf-pair")

Requires a CERN Kerberos ticket on the machine; obtain one with

    KRB5_CONFIG=/mnt/hephy/cms/Tools/krb5.conf kinit -fp schoef@CERN.CH

The files land under /eos/user/s/schoef/www/Jets/0070/<subdir> and are
served at https://schoef.web.cern.ch/schoef/Jets/0070/<subdir>/; an
index.php is copied alongside so the directory is browsable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EOS = "root://eosuser.cern.ch"
BASE = "/eos/user/s/schoef/www/Jets/0070"
URL = "https://schoef.web.cern.ch/schoef/Jets/0070"
INDEX_PHP = (
    "/users/robert.schoefbeck/CMS/ML/HEPHY-uncertainty/common/scripts/php/index.php"
)
KRB5_CONFIG = "/mnt/hephy/cms/Tools/krb5.conf"


def _environment() -> dict:
    environment = dict(os.environ)
    environment["KRB5_CONFIG"] = KRB5_CONFIG
    environment["CERN_USER"] = "schoef"
    return environment


def publish(paths, subdir: str) -> list[str]:
    """Copy the files to www/Jets/0070/<subdir>; return the public URLs."""

    environment = _environment()
    if subprocess.run(["klist", "-s"], env=environment).returncode != 0:
        raise RuntimeError(
            "no valid Kerberos ticket; run\n"
            f"  KRB5_CONFIG={KRB5_CONFIG} kinit -fp schoef@CERN.CH"
        )
    destination = f"{BASE}/{subdir}"
    subprocess.run(
        ["xrdfs", EOS, "mkdir", "-p", destination], env=environment, check=True
    )
    urls = []
    copies = [str(p) for p in paths]
    if Path(INDEX_PHP).exists():
        copies.append(INDEX_PHP)
    for path in copies:
        name = Path(path).name
        subprocess.run(
            ["xrdcp", "-f", "--nopbar", path, f"{EOS}//{destination.lstrip('/')}/{name}"],
            env=environment,
            check=True,
        )
        if name != "index.php":
            urls.append(f"{URL}/{subdir}/{name}")
    for url in urls:
        print(url)
    return urls


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python -m applications.cms.www <subdir> <files...>")
    publish(sys.argv[2:], sys.argv[1])


if __name__ == "__main__":
    main()

"""Offline bulk downloader for Kepler long-cadence light curves from MAST.

Fetches FITS directly from the static archive -- reading each target's directory
index for the exact per-quarter filenames -- rather than via lightkurve's slow
per-target search. Cached on disk and resumable; standard library only.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import shutil
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable

from absl import logging

_ARCHIVE = "https://archive.stsci.edu/missions/kepler/lightcurves"


def _target_dir_url(kepid: int) -> str:
  """Returns the MAST directory URL for a target's light curves."""
  kic = f"{kepid:09d}"
  return f"{_ARCHIVE}/{kic[:4]}/{kic}/"


def list_long_cadence_files(kepid: int, timeout: float = 30.0) -> list[str]:
  """Lists the long-cadence FITS filenames available for a target.

  Args:
    kepid: Kepler identification number.
    timeout: Per-request timeout in seconds.

  Returns:
    Sorted unique ``kplr<kic>-<timestamp>_llc.fits`` filenames (possibly empty).
  """
  request = urllib.request.Request(
      _target_dir_url(kepid), headers={"User-Agent": "hermes"}
  )
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      html = response.read().decode("utf-8", "ignore")
  except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
    return []
  return sorted(set(re.findall(rf"kplr{kepid:09d}-\d+_llc\.fits", html)))


def download_target(
    kepid: int, output_dir: str, timeout: float = 60.0, retries: int = 3
) -> list[str]:
  """Downloads all long-cadence FITS for one target, skipping cached files.

  Files are stored under ``output_dir/<kic9>/``.

  Args:
    kepid: Kepler identification number.
    output_dir: Root directory for the local FITS cache.
    timeout: Per-file timeout in seconds.
    retries: Number of download attempts per file.

  Returns:
    Local paths of the available (downloaded or cached) FITS files.
  """
  target_dir = os.path.join(output_dir, f"{kepid:09d}")
  os.makedirs(target_dir, exist_ok=True)
  base_url = _target_dir_url(kepid)

  paths: list[str] = []
  for filename in list_long_cadence_files(kepid):
    destination = os.path.join(target_dir, filename)
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
      paths.append(destination)
      continue
    if _download_file(base_url + filename, destination, timeout, retries):
      paths.append(destination)
  return paths


def download_targets(
    kepids: Iterable[int],
    output_dir: str,
    max_workers: int = 4,
    timeout: float = 60.0,
) -> dict[int, list[str]]:
  """Downloads light curves for many targets concurrently.

  Args:
    kepids: Kepler identification numbers (duplicates are de-duplicated).
    output_dir: Root directory for the local FITS cache.
    max_workers: Number of concurrent download workers.
    timeout: Per-file timeout in seconds.

  Returns:
    A dict mapping each ``kepid`` to its list of local FITS paths.
  """
  unique = sorted(set(int(k) for k in kepids))
  results: dict[int, list[str]] = {}
  total = len(unique)
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {
        pool.submit(download_target, kepid, output_dir, timeout): kepid
        for kepid in unique
    }
    for done, future in enumerate(
        concurrent.futures.as_completed(futures), start=1
    ):
      results[futures[future]] = future.result()
      if done % 25 == 0 or done == total:
        obtained = sum(1 for paths in results.values() if paths)
        logging.info(
            "downloaded %d/%d stars (%d with data)", done, total, obtained
        )
  return results


def _download_file(
    url: str, destination: str, timeout: float, retries: int
) -> bool:
  """Downloads a single file atomically, returning whether it succeeded.

  Streams the response with an enforced per-read timeout (unlike
  ``urllib.request.urlretrieve``, which ignores timeouts and can hang
  indefinitely on a stalled connection).
  """
  temporary = destination + ".part"
  for _ in range(retries):
    try:
      request = urllib.request.Request(url, headers={"User-Agent": "hermes"})
      with urllib.request.urlopen(request, timeout=timeout) as response:
        with open(temporary, "wb") as handle:
          shutil.copyfileobj(response, handle)
      os.replace(temporary, destination)
      return True
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
      if os.path.exists(temporary):
        os.remove(temporary)
      continue
  return False

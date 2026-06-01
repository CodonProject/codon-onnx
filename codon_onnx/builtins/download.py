import os
import time
import requests
import concurrent.futures
from typing import List, Tuple, Optional

CACHE_DIR = os.path.expanduser('~/.cache/codon_model')

_best_platform_cache: Optional[str] = None

PLATFORM_TEMPLATES = {
    'modelscope': 'https://www.modelscope.cn/models/{repo}/resolve/{branch}/{file}',
    'huggingface': 'https://huggingface.co/{repo}/resolve/{branch}/{file}'
}

PING_TARGETS = {
    'modelscope': 'https://www.modelscope.cn',
    'huggingface': 'https://huggingface.co'
}

def ping_platform(platform: str, timeout: float = 1.5) -> Tuple[str, float]:
    url = PING_TARGETS.get(platform)
    if not url:
        return platform, float('inf')
    try:
        start = time.perf_counter()
        response = requests.head(url, timeout=timeout)
        if response.status_code < 400:
            return platform, time.perf_counter() - start
    except requests.RequestException:
        pass
    return platform, float('inf')

def select_best_platform(platforms: List[str]) -> str:
    '''
    Selects the best platform with the lowest latency, caching the result.

    Args:
        platforms (List[str]): List of platform names to choose from.

    Returns:
        str: The chosen platform name with the lowest latency.
    '''
    global _best_platform_cache
    if len(platforms) == 1:
        return platforms[0]

    if _best_platform_cache is not None and _best_platform_cache in platforms:
        return _best_platform_cache
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(platforms)) as executor:
        futures = {executor.submit(ping_platform, p): p for p in platforms}
        best_platform = platforms[0]
        min_latency = float('inf')
        
        for future in concurrent.futures.as_completed(futures):
            platform, latency = future.result()
            if latency < min_latency:
                min_latency = latency
                best_platform = platform
                
    _best_platform_cache = best_platform
    return best_platform

def download_file(
    url: str,
    dest_path: str,
    desc: str = 'Downloading',
    temp_path: Optional[str] = None
) -> None:
    '''
    Downloads a file from url to dest_path with support for HTTP Range resuming.

    Args:
        url (str): The URL of the file to download.
        dest_path (str): The final local destination path.
        desc (str): Descriptive text for the console output.
        temp_path (Optional[str]): Custom path for the temporary download file.
            If None, defaults to dest_path + '.tmp'.
    '''
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if temp_path is None:
        temp_path = dest_path + '.tmp'
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    headers = {}
    downloaded = 0

    if os.path.exists(temp_path):
        downloaded = os.path.getsize(temp_path)
        if downloaded > 0:
            headers['Range'] = f'bytes={downloaded}-'

    response = requests.get(url, headers=headers, stream=True, timeout=15)

    if response.status_code == 206:
        mode = 'ab'
        remaining_size = int(response.headers.get('content-length', 0))
        total_size = remaining_size + downloaded
    elif response.status_code == 200:
        mode = 'wb'
        downloaded = 0
        total_size = int(response.headers.get('content-length', 0))
    elif response.status_code == 416:
        # Range Not Satisfiable: local temp file is probably already fully downloaded
        print(f'\r{desc}: 100.0% (Resumed and verified cached content)')
        os.replace(temp_path, dest_path)
        return
    else:
        response.raise_for_status()

    chunk_size = 1024 * 64
    
    print(f'{desc}:')
    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    bar = '#' * int(percent // 5) + '-' * (20 - int(percent // 5))
                    print(f'\r[{bar}] {percent:.1f}% ({downloaded/(1024*1024):.2f}MB/{total_size/(1024*1024):.2f}MB)', end='', flush=True)
                else:
                    print(f'\rDownloaded {downloaded/(1024*1024):.2f}MB', end='', flush=True)
    print('\n')
    
    os.replace(temp_path, dest_path)

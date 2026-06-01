import os
from typing import Literal, Optional, Dict, Any, TypeVar
from codon_onnx.builtins.download import CACHE_DIR, PLATFORM_TEMPLATES, select_best_platform, download_file


TRemoteResource = TypeVar('TRemoteResource', bound='RemoteResourceMixin')

class RemoteResourceMixin:
    __modelscope__: Optional[Dict[str, Any]] = None
    __huggingface__: Optional[Dict[str, Any]] = None
    __remote_resource__: Optional[Dict[str, Any]] = None

    def from_remote(
        self: TRemoteResource,
        platform: Optional[Literal['modelscope', 'huggingface']] = None, 
        url: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> TRemoteResource:
        base_cache_path = cache_dir or CACHE_DIR
        
        if url:
            filename = url.split('/')[-1]
            local_dir = os.path.join(base_cache_path, 'custom_downloads')
            local_path = os.path.join(local_dir, filename)
            
            if not os.path.exists(local_path):
                download_file(url, local_path, desc=f'Downloading custom resource')
            
            if hasattr(self, '_load_remote'):
                self._load_remote([local_path])
            elif hasattr(self, 'load_pretrained'):
                self.load_pretrained(local_path)
            elif hasattr(self, 'load'):
                self.load(local_path)
                
            return self

        available_sources = {}
        remote_resource = getattr(self, '__remote_resource__', None)

        if getattr(self, '__modelscope__', None):
            available_sources['modelscope'] = self.__modelscope__
        elif remote_resource:
            available_sources['modelscope'] = remote_resource

        if getattr(self, '__huggingface__', None):
            available_sources['huggingface'] = self.__huggingface__
        elif remote_resource:
            available_sources['huggingface'] = remote_resource
            
        if not available_sources:
            raise ValueError(f'Neither __modelscope__ nor __huggingface__ nor __remote_resource__ configuration was found in {self.__class__.__name__}.')

        if platform is None:
            cached_platforms = []
            for plat, config in available_sources.items():
                repo = config['repo']
                files = config.get('files', [])
                branch = config.get('branch', 'master' if plat == 'modelscope' else 'main')
                repo_subdir = repo.replace('/', '_')
                local_dir = os.path.join(base_cache_path, plat, repo_subdir, branch)
                if files:
                    has_all_files = all(os.path.exists(os.path.join(local_dir, f)) for f in files)
                    if has_all_files:
                        cached_platforms.append(plat)

            if cached_platforms:
                chosen_platform = select_best_platform(cached_platforms)
                print(f'[*] Using cached platform: {chosen_platform}')
            else:
                chosen_platform = select_best_platform(list(available_sources.keys()))
                print(f'[*] Auto-detected optimal platform: {chosen_platform}')
        else:
            chosen_platform = platform
            if chosen_platform not in available_sources:
                raise ValueError(f"Platform '{chosen_platform}' configuration is missing in this class.")

        local_paths = []
        
        for file in files:
            # Detect starting platform and initialize state
            current_platform = chosen_platform
            attempted_platforms = {current_platform}
            
            while True:
                config = available_sources[current_platform]
                repo = config['repo']
                files = config.get('files', [])
                branch = config.get('branch', 'master' if current_platform == 'modelscope' else 'main')

                repo_subdir = repo.replace('/', '_')
                local_dir = os.path.join(base_cache_path, current_platform, repo_subdir, branch)
                os.makedirs(local_dir, exist_ok=True)
                
                local_path = os.path.join(local_dir, file)
                shared_temp_path = os.path.join(base_cache_path, 'temp', repo_subdir, file + '.tmp')

                if os.path.exists(local_path):
                    if local_path not in local_paths:
                        local_paths.append(local_path)
                    break

                download_url = PLATFORM_TEMPLATES[current_platform].format(
                    repo=repo,
                    branch=branch,
                    file=file
                )
                
                try:
                    download_file(
                        download_url,
                        local_path,
                        desc=f'Retrieving {file} ({current_platform})',
                        temp_path=shared_temp_path
                    )
                    if local_path not in local_paths:
                        local_paths.append(local_path)
                    break
                except Exception as e:
                    print(f'\n[!] Error downloading from {current_platform}: {e}')
                    alt_platforms = [p for p in available_sources.keys() if p not in attempted_platforms]
                    if not alt_platforms:
                        raise RuntimeError(f"Failed to download '{file}' from all available platforms.") from e

                    next_platform = alt_platforms[0]
                    print(f'[*] Switching downloading platform from {current_platform} to {next_platform} for resume...')
                    current_platform = next_platform
                    attempted_platforms.add(next_platform)

        if hasattr(self, '_load_remote'):
            self._load_remote(local_paths)
        else:
            if local_paths:
                target_file = local_paths[0]
                if hasattr(self, 'load_pretrained'):
                    self.load_pretrained(target_file)
                elif hasattr(self, 'load'):
                    self.load(target_file)
                else:
                    raise NotImplementedError(f'No loader method found in {self.__class__.__name__}.')

        return self

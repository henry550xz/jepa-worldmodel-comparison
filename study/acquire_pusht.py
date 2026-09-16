"""Official OSF PushT archive: bounded parallel ranges, SHA verification, safe extraction."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath
from study.run import workspace_root

URL='https://osf.io/download/k2d8w/'
SIZE=2785304515
SHA256='442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08'


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True)
    p.add_argument('--connections',type=int,default=8);a=p.parse_args()
    if not 1<=a.connections<=8:raise ValueError('connections must be 1..8')
    root=workspace_root(a.workspace)/'datasets'; downloads=root/'downloads'; downloads.mkdir(exist_ok=True)
    archive=downloads/'pusht_noise.zip'
    start=time.monotonic()
    if not archive.exists():
        chunk=32*1024*1024
        ranges=[(i,min(i+chunk,SIZE)-1) for i in range(0,SIZE,chunk)]
        def fetch(pair):
            lo,hi=pair;path=downloads/f'pusht-{lo:012d}.part'
            if path.exists() and path.stat().st_size==hi-lo+1:return path
            subprocess.run(['curl','-L','--fail','--silent','--show-error','--connect-timeout','10',
                '--max-time','300','--speed-limit','65536','--speed-time','45','--retry','1',
                '--range',f'{lo}-{hi}','-o',str(path),URL],check=True)
            if path.stat().st_size!=hi-lo+1:raise RuntimeError('server ignored byte range')
            print(f'completed range {lo}-{hi}',flush=True)
            return path
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.connections) as pool:
            parts=list(pool.map(fetch,ranges))
        temporary=archive.with_suffix('.zip.part')
        with temporary.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:shutil.copyfileobj(src,out,1024*1024)
        temporary.rename(archive)
    digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if digest!=SHA256 or archive.stat().st_size!=SIZE:raise RuntimeError('official archive hash/size mismatch')
    with zipfile.ZipFile(archive) as z:
        total=sum(i.file_size for i in z.infolist())
        print('verified archive',SHA256,'uncompressed_bytes',total,flush=True)
        if shutil.disk_usage(root).free < total+5*1024**3:raise RuntimeError('insufficient extraction space plus reserve')
        for i in z.infolist():
            path=PurePosixPath(i.filename)
            if path.is_absolute() or '..' in path.parts or (i.external_attr >>16)&0o170000==0o120000:
                raise RuntimeError('unsafe archive entry')
        destination=root/'pusht_noise'
        if destination.exists():raise RuntimeError('refusing to overwrite existing dataset')
        z.extractall(root)
    # Archive itself preserves the official bytes; chunks are redundant owned temporary files.
    for part in downloads.glob('pusht-*.part'):part.unlink()
    inventory={'source':URL,'archive_sha256':digest,'archive_bytes':SIZE,
        'uncompressed_bytes':total,'dataset_path':str(root/'pusht_noise'),
        'download_and_extract_wall_seconds':time.monotonic()-start}
    (root/'PUSHT_DATASET.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory),flush=True)

if __name__=='__main__':main()

import os

IMG_EXTENSIONS=[
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp', '.npy', '.npz'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_rec(dir,images):
    assert os.path.isdir(dir),'%s is not a valid direction' %dir
    for root,dnames,fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path=os.path.join(root,fname)
                images.append(path)

def make_dataset(dir,recursive=False,read_cache=False,write_cache=False):
    images=[]
    if read_cache:
        possible_filelist=os.path.join(dir,'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist,'r') as f:
                images=f.read().splitlines()
                return images
    if recursive:
        make_dataset_rec(dir,images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir),'%s is not a valid directory'%dir
        for root,dnames,fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path=os.path.join(root,fname)
                    images.append(path)
    if write_cache:
        filelist_cache=os.path.join(dir,'files.list')
        with open(filelist_cache,'w') as f:
            for path in images:
                f.write("%s\n"%path)
            print("wrote filelist cache at %s"%filelist_cache)
    return images
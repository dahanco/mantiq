import os, zipfile, hashlib

for name in ("Mantiq-Portable-CPU", "Mantiq-Portable-CUDA"):
    root = rf"C:\tmp3\dist\{name}"
    zpath = rf"C:\tmp3\dist\{name}.zip"
    z = zipfile.ZipFile(zpath)
    entries = z.namelist()
    dirs = [e for e in entries if e.endswith("/")]
    files = [e for e in entries if not e.endswith("/")]
    bad_sep = [e for e in entries if "\\" in e]
    bad_prefix = [e for e in entries if not e.startswith(name + "/")]
    # files on disk vs files in the zip
    disk = set()
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            disk.add(os.path.relpath(os.path.join(dp, f), root).replace("\\", "/"))
    inzip = {e[len(name) + 1:] for e in files}
    print(f"== {name}")
    print(f"   entries={len(entries)} files={len(files)} dirs={len(dirs)}")
    print(f"   backslash={len(bad_sep)} bad_prefix={len(bad_prefix)}")
    print(f"   only-in-zip={sorted(inzip - disk)}")
    print(f"   only-on-disk={sorted(disk - inzip)}")
    version = z.read(name + "/VERSION").decode("utf-8").strip()
    print(f"   VERSION={version!r}")
    for key in ("mntq.py", "setup.py", "serve/server.py", "START-MNTQ.bat",
                "build-cpu/mntq-cpu.exe", "build-cpu/MNTQ-BUILD.json",
                "build/mntq.exe", "build/MNTQ-BUILD.json"):
        p = name + "/" + key
        print(f"   {key:<28} {'yes' if p in entries else '-'}")
    # embedded engine version must match the package's VERSION
    for exe in ("build-cpu/mntq-cpu.exe", "build/mntq.exe"):
        p = name + "/" + exe
        if p in entries:
            d = z.read(p)
            has = version.encode() in d
            print(f"   {exe}: size={len(d)} has{version}={has}")
            if not has:
                print(f"   ERROR: {exe} does not contain the declared version {version}")
    print("   sha256 =", hashlib.sha256(open(zpath, "rb").read()).hexdigest().upper())
    z.close()

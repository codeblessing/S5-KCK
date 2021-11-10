
# Score

Simple note recognition software based on image transformations.


## Development

### Prerequisites

* [Python][1] >= 3.9
* [Git][2]
* [Rustc & cargo][3] (only when compiling native modules)

### Setup

Linux:
```bash
git clone https://github.com/codeblessing/S5-KCK.git score
cd score
git checkout score
cd ./#05
py -m venv --upgrade-deps .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (with admin Powershell):
```powershell
git clone https://github.com/codeblessing/S5-KCK.git score
cd score
git checkout score
cd ./#05
py -m venv --upgrade-deps .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Now you're ready to go. \
[VS Code][4] is recommended IDE to work with.

[1]: https://python.org
[2]: https://git-scm.org
[3]: https://rust-lang.org/tools/install
[4]: https://code.visualstudio.com
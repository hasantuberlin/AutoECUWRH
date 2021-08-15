import pip
from subprocess import call
with open("requirements.txt") as myFile:
  pkgs = myFile.read()
  pkgs = pkgs.splitlines()
for pkg in pkgs:
    call("pip3 install " + pkg.split('==')[0], shell=True)
    
## BLAS and LAPACK

```LIBS=-framework Accelerate```.  Read ```man Accelerate``` for more information.

## Homebrew

- http://brew.sh/
- http://blog.shvetsov.com/2014/11/homebrew-cheat-sheet-and-workflow.html
- http://www.commandlinefu.com/commands/view/4831/update-all-packages-installed-via-homebrew
```
brew update && brew install `brew outdated`
```

## Macports

**I abandoned Macports completely after it switched my ABI to i386 without asking and broke absolutely everything horribly.  Now I use Homebrew instead.**

- http://superuser.com/questions/80168/how-to-uninstall-all-unused-versions-of-a-macports-package-at-once
- http://stackoverflow.com/questions/10319314/prevent-macports-from-installing-pre-built-package (in particular <tt>port -s -v install package +options</tt>)

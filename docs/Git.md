# Documentation and Useful Tips

* http://git-scm.com/book
* https://git.wiki.kernel.org/index.php/GitDocumentation
* http://git.or.cz/course/svn.html
* http://www.youtube.com/GitHubGuides
* http://jnrbsn.com/2010/11/how-to-mirror-a-subversion-repository-on-github
* http://stackoverflow.com/questions/296975/how-do-i-tell-git-svn-about-a-remote-branch-created-after-i-fetched-the-repo
* http://ivanz.com/2009/01/15/selective-import-of-svn-branches-into-a-gitgit-svn-repository/
* https://www.kernel.org/pub/software/scm/git/docs/git-clean.html
* http://stackoverflow.com/questions/661018/pushing-an-existing-git-repository-to-svn
* http://nuclearsquid.com/writings/git-tricks-tips-workflows/
* http://tom.preston-werner.com/2009/05/19/the-git-parable.html
* http://git-scm.com/book/en/Git-Tools-Debugging-with-Git

# Visualization

* https://github.com/esc/git-big-picture
* https://code.google.com/p/gitgraph/

# ALCF Install 

```
git clone https://github.com/git/git
cd git
git checkout tags/v1.8.3.4
make prefix=/soft/versioning/git/1.8.3.4 all NO_OPENSSL=YesPlease NO_EXPAT=YesPlease 
make prefix=/soft/versioning/git/1.8.3.4 install NO_OPENSSL=YesPlease NO_EXPAT=YesPlease 
```

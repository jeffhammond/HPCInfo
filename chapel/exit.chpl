proc main() {
    if (numLocales<2) {
        writeln("You need to run with >1 locale");
        //exit;  // bad
        exit(0); // good
    }
}

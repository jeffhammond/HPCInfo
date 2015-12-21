proc main() {
    if (numLocales<2) {
        writeln("You need to run with >1 locale");
        exit(1);
    }
    coforall loc in Locales do
        on Locales(0) {
        }
        on Locales(1) {
        }
}

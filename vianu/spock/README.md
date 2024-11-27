# Spotting Clinical Knowledge (SpoCK)

## CLI
```bash
alias spock="python -m vianu.spock"
```
### Scraping
```bash
spock scraping --source pubmed --term dafalgan
```

### Chunking
```bash
spock chunking --min-chunk-size 30 --min-chunk-overlap 5 --text-file "tests/tests-spock/data/chunker.txt"
```

### Pipeline
```bash
spock pipeline --steps "all" --min-chunk-size 30 --min-chunk-overlap 5 --text-file "tests/tests-spock/data/chunker.txt" --source pubmed --term dafalgan
spock pipeline --steps "scp,cnk"
```

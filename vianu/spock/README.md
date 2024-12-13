# Spotting Clinical Knowledge (SpoCK)

## CLI
```bash
alias spock="python -m vianu.spock"
```
### Scraping
```bash
spock pipeline --steps "scp" --source pubmed --term dafalgan
```

### Chunking

TODO define a file and load it from there with --data-load
```bash
spock chunking --min-chunk-size 30 --min-chunk-overlap 5 --text-file "tests/tests-spock/data/chunking.txt"
```

### Pipeline
```bash
spock pipeline --steps "all" --min-chunk-size 30 --min-chunk-overlap 5 --source pubmed --term dafalgan
spock pipeline --steps "scp,cnk" TODO
```

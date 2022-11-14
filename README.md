# `unfold`
**UN**packing **F**or scenari**O**-based **L**ca **D**atabases

## What does `unfold` do?

1. it unfolds data packages containing scenario data (modified exchange values as well as additional inventories),
2. builds scenario-specific ecoinvent databases,
3. and registers them into a `brightway` project.

`unfold` is initially conceived to share `premise`-generated databases, without sharing the underlying data which is under restrictive licensing (i.e., ecoinvent).

## Limitations

* only works with `brightway2` at the moment


## How to

### Install

```bash
pip install unfold
```

### Use

    import unfold
    
    u = Unfold('path/to/datapackage.zip')
    u.unfold()
    u.build()
    u.write()

## Author

[Romain Sacchi](romain.sacchi@psi.ch), PSI

## License

See [License](https://github.com/romainsacchi/stunt/blob/main/LICENSE).

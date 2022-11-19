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

`unfold` is available on PyPI and can be installed with `pip`:

    pip install unfold


### Use

#### fold

``unfold`` can "fold" several brightway2 databases
into a single data package. The data package is a zip file
containing the differences of the databases in relation
to a source database (including extra inventories), 
as well as a metadata file that describes the databases 
and their content.

    from unfold import Fold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to fold
    bw2data.projects.set_current("some BW2 project")
    
    f = Fold()
    f.fold()

#### unfold

``unfold`` can "unfold" a data package into one or several 
brightway2 databases.

    from unfold import Unfold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to unfold
    bw2data.projects.set_current("some BW2 project")
    
    u = Unfold("a package name.zip")
    u.unfold()

#### unfold a superstructure database (to be used with Activity Browser)

``unfold`` can "unfold" a data package into a superstructure database
that can be used with the Activity Browser.

    from unfold import Unfold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to unfold
    bw2data.projects.set_current("some BW2 project")
    
    u = Unfold("a package name.zip")
    u.unfold(superstructure=True)


## Author

[Romain Sacchi](mailto:romain.sacchi@psi.ch), PSI

## License

See [License](https://github.com/romainsacchi/stunt/blob/main/LICENSE).

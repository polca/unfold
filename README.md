# `unfold`

<p align="center">
  <a href="https://badge.fury.io/py/unfold" target="_blank"><img src="https://badge.fury.io/py/unfold.svg"></a>
  <a href="https://github.com/romainsacchi/unfold" target="_blank"><img src="https://github.com/romainsacchi/unfold/actions/workflows/main.yml/badge.svg?branch=main"></a>
</p>

Publicly share LCA databases that are based on licensed data.

## What does `unfold` do?

``unfold`` is a Python package that allows "folding" and "unfolding"
LCA databases derived from a source database (e.g., ecoinvent) without
exposing the data contained in the source database.

![flow diagram](assets/flow_diagram.png)

The purpose of this package is to allow users to publicly share 
LCA databases without sharing the source database, in the case
where the latter is under restrictive license. Hence, ```unfold``` 
allows users to share instead data packages that allows other users
to reproduce the LCA database (provided they have the source database).

It is based on the [brightway2](https://brightway.dev) framework.

`unfold` is initially conceived to share `premise`-generated 
databases ([link](https://github.com/polca/premise)), without sharing the underlying data which is under 
restrictive licensing (i.e., ecoinvent).

## Limitations

* only works with `brightway2` at the moment


## How to

### Install

`unfold` is available on PyPI and can be installed with `pip`:

    pip install unfold


### Use

See also examples notebooks in the `examples` folder.

#### fold

``unfold`` can "fold" several brightway2 databases
into a single data package. The data package is a zip file
containing the differences of the databases in relation
to a source database (including extra inventories), 
as well as a metadata file that describes the databases 
and their content.

```python

    from unfold import Fold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to fold
    bw2data.projects.set_current("some BW2 project")
    
    f = Fold()
    f.fold()
```

#### unfold

``unfold`` can "unfold" a data package into one or several 
brightway2 databases.

```python

    from unfold import Unfold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to unfold
    bw2data.projects.set_current("some BW2 project")
    
    u = Unfold("a package name.zip")
    u.unfold()
```

#### unfold a superstructure database (to be used with Activity Browser)

``unfold`` can "unfold" a data package into a superstructure database
that can be used with the Activity Browser.

```python

    from unfold import Unfold
    import bw2data
    
    # name of the brightway project containing 
    # both the source database and the databases to unfold
    bw2data.projects.set_current("some BW2 project")
    
    u = Unfold("a package name.zip")
    u.unfold(superstructure=True)
```

## Author

[Romain Sacchi](mailto:romain.sacchi@psi.ch), PSI

## License

See [License](https://github.com/romainsacchi/stunt/blob/main/LICENSE).

---
title: '`unfold`: removing the barriers to sharing and reproducing prospective life-cycle assessment databases'
tags:
  - Python
  - life cycle assessment
  - prospective
  - database
  - premise
  - scenario

authors:
  - name: Romain Sacchi
    orcid: 0000-0003-1440-0905
    affiliation: 1

affiliations:
 - name: Paul Scherrer Institute, Villigen, Switzerland
   index: 1

date: 05 January 2023
bibliography: paper.bib

---

# Summary

`unfold` is a Python package that allows reproducing life-cycle databases which
partially build on a data source that cannot be shared. It produces 
data packages that contain the differences between the databases to share and 
the licensed data source. The data package also contains a metadata file that 
describes the databases, the author, and other useful information. `unfold` allows to 
pack and unpack any number of databases from a single data package, to ease the 
sharing and reproducibility of prospective or scenario-based life-cycle assessment 
databases.

# Statement of need

Life-cycle assessment (LCA) consists in the quantification of the environmental 
footprint of a product or a process by considering impacts along each of its 
relevant life-cycle phases [@ISO:2006]. It relies on databases describing 
thousands of industrial processes, which interdependency and interactions 
with the greater environment form large input-output matrices. 
However, these matrices, often modified to fit the need of a study, 
are difficult to share when they contain data covered by a restrictive 
end-user license agreement, such as that of the ecoinvent database [@Wernet:2016].

The difficulty to share these databases prevents the reproducibility of results 
and impedes the scientific validity of the work produced by the LCA community.

Using the LCA framework `brightway2` [@Mutel:2017], `unfold` allows to share LCA 
databases without exposing the data under license, sharing instead data packages used 
to reproduce the databases locally (provided the end-users also have access to the 
licensed data source).

`unfold` is initially conceived to share LCA databases that have been heavily 
modified, such as those generated with the premise package [@Sacchi:2022], or 
those regularly produced within the field of prospective LCA (where the need to 
modify extensively the source database is often required) – see the work of 
Cox [@Cox:2018], Mendoza [@Mendoza:2018], Joyce [@Joyce:2022] and colleagues.

# Description

LCA relies on a set of input-output matrices representing product and service exchanges
that are typically sparse, with a density inferior to 1%. Sometimes, for the need of sensitivity
or scenario analysis, up to hundreds of thousands of exchanges are added, modified, 
or removed from the technosphere matrix, potentially reflecting deep changes in certain 
industrial sectors. [@Mendoza:2018] gives an example, where exchanges 
related to power generation are modified throughout the database to reflect changes 
in generation efficiencies, regional electricity mixes, etc. 

`unfold` calculates the scaling factors for each modified exchange with 
reference to its value prior modification (i.e., the reference database) and stores 
them into a data package. 

When the data package is read by another user, the scaling factors are multiplied 
by the exchange value found in the local reference database to back-calculate 
the value of the modified exchange.

## Software architecture

The example illustrated in Figure 1 represents a use case where User 1 (in blue) 
would like to share databases A and B, which are initially derived from database C 
– which cannot be shared. `unfold` creates a data package (zip file) where the 
difference across exchanges in databases A and B relative to the reference 
database C are stored as scaling factors. Additional metadata (e.g., author, 
description) is included in the data package, as per the specifications given by 
Frictionless [@Walsh:2022]. User 2 (in green) reads the data package using 
`unfold` on its local computer, which back-calculates the value of each exchange 
by multiplying the original values found in the reference database with their 
corresponding scaling factors. This assumes User 2 also has local access 
to the reference database C.

![Workflow for sharing databases using `unfold` data packages.\label{fig:workflow}](assets/flow_diagram.png)

## Software functionalities

`unfold` offers two functions: pack and unpack LCA databases. The packing 
function calculates the differences between one or several modified databases 
and a reference database and stores them into a data package, ready to be shared. 
The unpacking function creates and registers one or several LCA databases by 
reading the data package and the local reference database.

`unfold` can also produce superstructure databases [@Steubing:2021], which can integrate 
several scenarios into a single `brightway2` database, to be further used with the LCA 
software `activity-browser` [@Steubing:2020].

# Impact

`unfold` allows sharing LCA databases that partially build on proprietary data 
without exposing the data under license. This eases the process of 
reproducibility, which is an important prerequisite to scientific validity. 

Doing so allows sharing the work of the different authors of these databases, 
to be reused within the community. To that effect, a repository that lists 
several data packages already exists [@Sacchi:2022b], permitting users to 
reproduce and use prospective LCA databases without having to execute 
the steps, or operate the tools, that led to their generation.

# Conclusions

`unfold` offers a way to exchange and reproduce LCA databases that build partially 
on licensed data through the sharing of scaling factors in a data package.

# Acknowledgements

Financial support was provided by the Kopernikus Project Ariadne (FKZ 03SFK5A) 
funded by the German Federal Ministry of Education and Research


# References
# COGANT Bridge

The bridge package is the project-local miniature of the COGANT idea: structured
system facts become generative-model priors, and a live engine can be re-emitted
as a GNN-style summary [5, 6]. COGANT is the translation design constraint this file
adapts locally.

## Firm structure to priors

`firm_structure_to_channels()` accepts a mapping of firm-structure counts and
returns channel priors. The aliases are intentionally business-readable:

| Channel | Accepted count keys |
| --- | --- |
| `I` | `book_size`, `book`, `aum` |
| `S` | `data_feeds`, `feeds` |
| `U` | `venues`, `routes` |
| `Theta` | `models`, `ewm_models` |
| `Z` | `researchers`, `research_staff` |

Counts must be finite and non-negative. The count is transformed through a
logistic-scaled `log1p` map so priors move smoothly rather than jumping.

## Round trip

`model_to_gnn_summary()` emits:

- `StateSpaceBlock`
- `Connections`
- `InitialParameterization`
- `ActInfOntologyAnnotation`
- footer metadata

`parse_gnn_summary()` parses the emitted text back into factors, connections,
and ontology annotations. The round trip is structural, not byte-for-byte. Its
purpose is to prove that the executable `EconomicWorldModel` preserves the GNN
sections needed by the manuscript and validators [5].

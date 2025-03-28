# Overview

In Manuscript v2, we implemented necessary improvements to our Summarizer. All relevant textual data involved in this improvement are provided in this file dir.

## Data Sources

### Orphanet Dataset (`DeepSeek_R1_671B_Response_Orphanet.csv`)

| Column | Description |
| --- | --- |
| **Orphanet_ID** | Unique identifier assigned by Orphanet for diseases |
| **Definition** | Disease definition text directly sourced from the 'Definition' section of Orphanet disease entries |
| **HPO_Definition** | Concatenated text of definitions from all Human Phenotype Ontology (HPO) terms annotated to the disease |
| **Related_HPO_Terms** | Specific HPO terminology associated with the disease |
| **Extracted_Content** | Text output from DeepSeek-R1-671B, generated using input format: `{Prompt + HPO_Description}` |

### OMIM Dataset (`DeepSeek_R1_671B_Response_OMIM.csv`)

| Column | Description |
| --- | --- |
| **OMIM_ID** | Unique identifier from Online Mendelian Inheritance in Man (OMIM) database for genetic disorders |
| **Description** | Disease description text sourced from OMIM official entries |
| **HPO_Description** | Concatenated definitions of all HPO terms annotated to the disease |
| **Related_HPO_Terms** | Associated HPO terminology for the disease |
| **Extracted_content** | DeepSeek-R1-671B output text generated from `{Prompt + HPO_Description}` input |

##

## Bio-Medical-3B-CoT

The model parameters are available at:  
[ContactDoctor/Bio-Medical-3B-CoT-012025 · Hugging Face](https://huggingface.co/ContactDoctor/Bio-Medical-3B-CoT-012025)

## Additional Resources

All relevant training files and sample training code are provided in the associated folder. For complete methodological details, please refer to our manuscript.

from __future__ import annotations


LAMANNO_LINEAGES: dict[str, set[str]] = {
    "lineage_1": {
        "Erythroid progenitor",
        "Erythrocyte",
        "Angioblast",
        "Endothelial",
    },
    "lineage_2": {
        "Epiblast",
        "Early ectoderm",
        "Neural crest",
        "Mesenchyme",
        "Early fibroblasts",
        "Intermediate meninges 1",
    },
    "lineage_3": {
        "Hindbrain",
        "Cerebellum glutamatergic",
        "Hindbrain roof plate",
    },
    "lineage_4": {
        "Dorsal forebrain",
        "Forebrain",
        "Ventral forebrain",
        "Neuronal intermediate progenitor",
        "Forebrain glutamatergic",
        "Forebrain GABAergic",
        "Cortical or hippocampal glutamatergic",
    },
}

LAMANNO_LINEAGE_CELLTYPE_ORDER: dict[str, tuple[str, ...]] = {
    "lineage_1": (
        "Erythroid progenitor",
        "Erythrocyte",
        "Angioblast",
        "Endothelial",
    ),
    "lineage_2": (
        "Epiblast",
        "Early ectoderm",
        "Neural crest",
        "Mesenchyme",
        "Early fibroblasts",
        "Intermediate meninges 1",
    ),
    "lineage_3": (
        "Hindbrain",
        "Hindbrain roof plate",
        "Cerebellum glutamatergic",
    ),
    "lineage_4": (
        "Dorsal forebrain",
        "Forebrain",
        "Ventral forebrain",
        "Neuronal intermediate progenitor",
        "Forebrain glutamatergic",
        "Forebrain GABAergic",
        "Cortical or hippocampal glutamatergic",
    ),
}

GROUP_1M = {
    "Cortical or hippocampal glutamatergic",
    "Forebrain glutamatergic",
    "Forebrain",
    "Forebrain GABAergic",
    "Dorsal forebrain",
    "Cajal-Retzius",
    "Forebrain astrocyte",
    "Thalamus glutamatergic",
    "Hypothalamus glutamatergic",
    "Hypothalamus",
    "Dorsal diencephalon",
    "Cortical hem",
    "Neuronal intermediate progenitor",
}

GROUP_2M = {
    "Midbrain glutamatergic",
    "Hindbrain glutamatergic",
    "Cerebellum glutamatergic",
    "Midbrain",
    "Dorsal midbrain",
    "Ventral midbrain",
    "Midbrain-hindbrain boundary",
    "Midbrain basal plate",
    "Hindbrain",
    "Dorsal hindbrain",
    "Hindbrain roof plate",
    "Epithalamus glutamatergic",
    "Hindbrain GABAergic",
    "Cerebellum GABAergic",
    "Spinal cord glutamatergic",
    "Spinal cord glycinergic",
    "Hindbrain glycinergic",
}

GROUP_3M = {
    "Midbrain GABAergic",
    "Mixed region GABAergic",
    "Mixed region and neurotransmitter",
    "Mixed region glutamatergic",
    "Mixed region",
    "Mixed region astrocytes",
    "Forebrain astrocyte",
    "Motor neuron",
    "Sensory neuron",
    "Caudal",
    "Anterior",
    "Pia 2",
    "Pia 3",
    "Optic cup",
}

GROUP_4M = {
    "Erythroid progenitor",
    "Erythrocyte",
    "Early fibroblasts",
    "Mesenchyme",
    "Mesoderm",
    "Nascent mesoderm",
    "Paraxial",
    "Endothelial",
    "Angioblast",
    "Pericyte",
    "Definitive endoderm",
    "Extraembryonic endoderm",
    "PreOPC",
    "Oligodendrocyte precursor cell",
    "Bergmann glia",
    "Early ectoderm",
    "Epiblast",
    "Neural crest",
    "Schwann cell",
    "Non-cycling microglia",
    "Axon tract-associated microglia",
    "Non-cycling perivascular macrophages",
    "Arachnoid",
    "Intermediate meninges 1",
    "Undefined",
}

LAMANNO_GROUPS_M = {
    "group_1m": GROUP_1M,
    "group_2m": GROUP_2M,
    "group_3m": GROUP_3M,
    "group_4m": GROUP_4M,
}

GROUP_1M_NEURONAL_FOREBRAIN = {
    "Cortical or hippocampal glutamatergic",
    "Forebrain glutamatergic",
    "Forebrain GABAergic",
    "Cajal-Retzius",
}

GROUP_1M_PROGENITOR_REGIONAL_DIENCEPHALON = {
    "Forebrain",
    "Dorsal forebrain",
    "Neuronal intermediate progenitor",
    "Cortical hem",
    "Forebrain astrocyte",
    "Thalamus glutamatergic",
    "Hypothalamus glutamatergic",
    "Hypothalamus",
    "Dorsal diencephalon",
}

LAMANNO_GROUPS_M_SPLIT = {
    "group_1m_neuronal_forebrain": GROUP_1M_NEURONAL_FOREBRAIN,
    "group_1m_progenitor_regional_diencephalon": GROUP_1M_PROGENITOR_REGIONAL_DIENCEPHALON,
    "group_2m": GROUP_2M,
    "group_3m": GROUP_3M,
    "group_4m": GROUP_4M,
}

# Hematopoietic / Endothelial lineage
groupl_1 = {
    "Erythroid progenitor",
    "Erythrocyte",
    "Angioblast",
    "Endothelial",
    "Platelet",
    "Hematoendothelial lineage",
    "Early macrophage",
    "Non-cycling perivascular macrophages",
    "Cycling perivascular macrophages",
    "Non-cycling microglia",
    "Cycling microglia",
    "Axon tract-associated microglia",
    "Infiltrating immune"
}

# Early embryonic / mesoderm / fibroblast lineage
groupl_2 = {
    "Epiblast",
    "Early ectoderm",
    "Surface ectoderm",
    "Neural crest",
    "Mesenchyme",
    "Early fibroblasts",
    "Intermediate meninges 1",
    "Intermediate meninges 2",
    "Perivascular fibroblast-like cells",
    "Nascent mesoderm",
    "Paraxial",
    "Mesoderm",
    "Anterior primitive streak",
    "Cardiac mesoderm",
    "Osteochondral mesoderm",
    "Pericyte",
    "Vascular smooth muscle",
    "Dura",
    "Arachnoid",
    "Arachnoid barrier cells"
}

# Endoderm / extraembryonic / gut lineage / # Glial / neural support lineage
groupl_3 = {
    "Endoderm",
    "Extraembryonic endoderm",
    "Extraembryonic ectoderm",
    "Definitive endoderm",
    "Visceral endoderm",
    "Gut",
    "Pharyngeal",
    "Pharyngeal epithelium",
    "Oral ectoderm",
    "Adenohypophyseal placode",
    "Pituitary neuroendocrine",
    "Oligodendrocyte",
    "Oligodendrocyte precursor cell",
    "PreOPC",
    "Committed oligodendrocyte precursor",
    "Ependymal-like",
    "Mixed region astrocytes",
    "Schwann cell",
    "Olfactory ensheathing cell",
    "Choroid plexus",
    "Early choroid plexus",
    "Pia 1",
    "Pia 2",
    "Pia 3"
}

# Neural progenitor / forebrain lineage
groupl_4 = {
    "Dorsal forebrain",
    "Forebrain",
    "Forebrain glutamatergic",
    "Forebrain GABAergic",
    "Forebrain astrocyte",
    "Cortical or hippocampal glutamatergic",
    "Cortical hem",
    "Cajal-Retzius",
    "Anteromedial cerebral pole",
    "Diencephalon",
    "Diencephalon glutamatergic",
    "Thalamus glutamatergic",
    "Hypothalamus",
    "Hypothalamus glutamatergic",
    "Hypothalamic floor-plate like",
    "Zona limitans intrathalamica",
    "Dorsal diencephalon",
    "Dorsal early ectoderm"
}

# Midbrain lineage
groupl_5 = {
    "Midbrain",
    "Dorsal midbrain",
    "Ventral midbrain",
    "Dorsal midbrain glutamatergic",
    "Midbrain dopaminergic",
    "Midbrain basal plate",
    "Midbrain floor plate",
    "Midbrain-hindbrain boundary",
    "Midbrain glutamatergic",
    "Midbrain GABAergic",
    "Midbrain-hindbrain GABAergic",
    "Midbrain floor plate"
}

# Hindbrain / cerebellum lineage / Spinal cord / motor neurons lineage
groupl_6 = {
    "Hindbrain",
    "Hindbrain glutamatergic",
    "Hindbrain GABAergic",
    "Hindbrain glycinergic",
    "Hindbrain serotoninergic",
    "Dorsal hindbrain",
    "Dorsal hindbrain and spinal cord",
    "Posterior hindbrain glutamatergic",
    "Hindbrain roof plate",
    "Fourth-ventricle roof plate",
    "Roof plate",
    "Cerebellum glutamatergic",
    "Cerebellum GABAergic",
    "Bergmann glia",
    "Spinal cord",
    "Spinal cord glutamatergic",
    "Spinal cord glycinergic",
    "Motor neuron",
    "Floor plate"
}


# Sensory / placodal / other neural structures / Miscellaneous / undefined
groupl_9 = {
    "Sensory neuron",
    "Otic vesicle",
    "Lens",
    "Olfactory pit",
    "Lateral nasal pit",
    "Optic cup",
    "Mixed region",
    "Mixed region glutamatergic",
    "Mixed region GABAergic",
    "Mixed region and neurotransmitter",
    "Undefined",
    "Node",
    "Primordial germ cells",
    "Subcommissural organ hypendymal cell",
    "Anteromedial cerebral pole"
}

# Combine into final dictionary
LAMANNO_GROUPS_L = {
    "groupl_1": groupl_1,
    "groupl_2": groupl_2,
    "groupl_3": groupl_3,
    "groupl_4": groupl_4,
    "groupl_5": groupl_5,
    "groupl_6": groupl_6,
    "groupl_9": groupl_9,
}

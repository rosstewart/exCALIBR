def get_acmg_evidence_codes():
    return {
        # Very strong evidence of pathogenicity
        "PVS1": "Pathogenic_Very_Strong",
        "PVS1_Supporting": "Pathogenic_Supporting",
        "PVS1_Moderate": "Pathogenic_Moderate",
        "PVS1_Strong": "Pathogenic_Strong",
        "PVS1_Very": "Pathogenic_Very_Strong",
    
        # Strong evidence of pathogenicity
        "PS1": "Pathogenic_Strong",
        "PS1_Supporting": "Pathogenic_Supporting",
        "PS1_Moderate": "Pathogenic_Moderate",
        "PS1_Very": "Pathogenic_Very_Strong",
    
        "PS2": "Pathogenic_Strong",
        "PS2_Supporting": "Pathogenic_Supporting",
        "PS2_Moderate": "Pathogenic_Moderate",
        "PS2_Very": "Pathogenic_Very_Strong",
    
        "PS3": "Pathogenic_Strong",
        "PS3_Supporting": "Pathogenic_Supporting",
        "PS3_Moderate": "Pathogenic_Moderate",
        "PS3_Very": "Pathogenic_Very_Strong",
    
        "PS4": "Pathogenic_Strong",
        "PS4_Supporting": "Pathogenic_Supporting",
        "PS4_Moderate": "Pathogenic_Moderate",
        "PS4_Very": "Pathogenic_Very_Strong",
    
        # Moderate evidence of pathogenicity
        "PM1": "Pathogenic_Moderate",
        "PM1_Supporting": "Pathogenic_Supporting",
        "PM1_Strong": "Pathogenic_Strong",
        "PM1_Very": "Pathogenic_Very_Strong",
    
        "PM2": "Pathogenic_Moderate",
        "PM2_Supporting": "Pathogenic_Supporting",
        "PM2_Strong": "Pathogenic_Strong",
        "PM2_Very": "Pathogenic_Very_Strong",
    
        "PM3": "Pathogenic_Moderate",
        "PM3_Supporting": "Pathogenic_Supporting",
        "PM3_Strong": "Pathogenic_Strong",
        "PM3_Very": "Pathogenic_Very_Strong",
    
        "PM4": "Pathogenic_Moderate",
        "PM4_Supporting": "Pathogenic_Supporting",
        "PM4_Strong": "Pathogenic_Strong",
        "PM4_Very": "Pathogenic_Very_Strong",
    
        "PM5": "Pathogenic_Moderate",
        "PM5_Supporting": "Pathogenic_Supporting",
        "PM5_Strong": "Pathogenic_Strong",
        "PM5_Very": "Pathogenic_Very_Strong",
    
        "PM6": "Pathogenic_Moderate",
        "PM6_Supporting": "Pathogenic_Supporting",
        "PM6_Strong": "Pathogenic_Strong",
        "PM6_Very": "Pathogenic_Very_Strong",
    
        # Supporting evidence of pathogenicity
        "PP1": "Pathogenic_Supporting",
        "PP1_Moderate": "Pathogenic_Moderate",
        "PP1_Strong": "Pathogenic_Strong",
        "PP1_Very": "Pathogenic_Very_Strong",
    
        "PP2": "Pathogenic_Supporting",
        "PP2_Moderate": "Pathogenic_Moderate",
        "PP2_Strong": "Pathogenic_Strong",
        "PP2_Very": "Pathogenic_Very_Strong",
    
        "PP3": "Pathogenic_Supporting",
        "PP3_Moderate": "Pathogenic_Moderate",
        "PP3_Strong": "Pathogenic_Strong",
        "PP3_Very": "Pathogenic_Very_Strong",
    
        "PP4": "Pathogenic_Supporting",
        "PP4_Moderate": "Pathogenic_Moderate",
        "PP4_Strong": "Pathogenic_Strong",
        "PP4_Very": "Pathogenic_Very_Strong",
    
        "PP5": "Pathogenic_Supporting",
        "PP5_Moderate": "Pathogenic_Moderate",
        "PP5_Strong": "Pathogenic_Strong",
        "PP5_Very": "Pathogenic_Very_Strong",
    
        # Benign standalone
        "BA1": "Benign_Standalone",
    
        # Strong evidence of benignity
        "BS1": "Benign_Strong",
        "BS1_Supporting": "Benign_Supporting",
        "BS1_Moderate": "Benign_Moderate",
        "BS1_Very": "Benign_Very_Strong",
    
        "BS2": "Benign_Strong",
        "BS2_Supporting": "Benign_Supporting",
        "BS2_Moderate": "Benign_Moderate",
        "BS2_Very": "Benign_Very_Strong",
    
        "BS3": "Benign_Strong",
        "BS3_Supporting": "Benign_Supporting",
        "BS3_Moderate": "Benign_Moderate",
        "BS3_Very": "Benign_Very_Strong",
    
        "BS4": "Benign_Strong",
        "BS4_Supporting": "Benign_Supporting",
        "BS4_Moderate": "Benign_Moderate",
        "BS4_Very": "Benign_Very_Strong",
    
        # Supporting evidence of benignity
        "BP1": "Benign_Supporting",
        "BP1_Moderate": "Benign_Moderate",
        "BP1_Strong": "Benign_Strong",
        "BP1_Very": "Benign_Very_Strong",
    
        "BP2": "Benign_Supporting",
        "BP2_Moderate": "Benign_Moderate",
        "BP2_Strong": "Benign_Strong",
        "BP2_Very": "Benign_Very_Strong",
    
        "BP3": "Benign_Supporting",
        "BP3_Moderate": "Benign_Moderate",
        "BP3_Strong": "Benign_Strong",
        "BP3_Very": "Benign_Very_Strong",
    
        "BP4": "Benign_Supporting",
        "BP4_Moderate": "Benign_Moderate",
        "BP4_Strong": "Benign_Strong",
        "BP4_Very": "Benign_Very_Strong",
    
        "BP5": "Benign_Supporting",
        "BP5_Moderate": "Benign_Moderate",
        "BP5_Strong": "Benign_Strong",
        "BP5_Very": "Benign_Very_Strong",
    
        "BP6": "Benign_Supporting",
        "BP6_Moderate": "Benign_Moderate",
        "BP6_Strong": "Benign_Strong",
        "BP6_Very": "Benign_Very_Strong",
    
        "BP7": "Benign_Supporting",
        "BP7_Moderate": "Benign_Moderate",
        "BP7_Strong": "Benign_Strong",
        "BP7_Very": "Benign_Very_Strong"
    }



# Define function to classify based on ACMG evidence codes
def classify_acmg(evidence_list):
   
    if not evidence_list:
        return "VUS"  # Default to Uncertain Significance if no evidence is left

    acmg_evidence_codes = get_acmg_evidence_codes()

    # Categorize evidence into strengths
    pathogenic_very_strong = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Pathogenic_Very_Strong")
    pathogenic_strong = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Pathogenic_Strong")
    pathogenic_moderate = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Pathogenic_Moderate")
    pathogenic_supporting = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Pathogenic_Supporting")

    benign_standalone = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == 'Benign_standalone')
    benign_strong = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Benign_Strong")
    benign_moderate = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Benign_Moderate")
    benign_supporting = sum(1 for e in evidence_list if acmg_evidence_codes.get(e) == "Benign_Supporting")

    # Pathogenic Classification Rules
    if pathogenic_very_strong >= 1:
        if pathogenic_strong >= 1 or pathogenic_moderate >= 2:
            return "Pathogenic"
    if pathogenic_strong >= 2:
        return "Pathogenic"
    if pathogenic_strong == 1 and pathogenic_moderate >= 2:
        return "Pathogenic"
    if pathogenic_very_strong == 1 and pathogenic_moderate >= 1:
        return "Likely Pathogenic"
    if pathogenic_strong == 1 and pathogenic_moderate == 1:
        return "Likely Pathogenic"
    if pathogenic_moderate >= 3:
        return "Likely Pathogenic"
    if pathogenic_moderate == 2 and pathogenic_supporting >= 2:
        return "Likely Pathogenic"

    # Benign Classification Rules
    if benign_standalone >= 1:
        return 'Benign'
    if benign_strong >= 2:
        return "Benign"
    if benign_strong == 1 and benign_supporting >= 1:
        return "Likely Benign"
    if benign_moderate >= 2:
        return "Likely Benign"

    # Default to VUS if no strong evidence
    return "VUS"
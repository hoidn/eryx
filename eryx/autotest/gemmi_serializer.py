# gemmi_serializer.py

import gemmi

class GemmiSerializer:
    """
    Utility for serializing and restoring Gemmi objects to/from Python dictionaries.
    This handles gemmi.Structure with basic details (cell, space group,
    per-model/chain/residues/atoms).
    
    Example usage:
        serializer = GemmiSerializer()
        
        # Serialize
        my_struct = gemmi.read_structure("example.pdb")
        serialized_dict = serializer.serialize_structure(my_struct)
        
        # Deserialize
        restored_struct = serializer.deserialize_structure(serialized_dict)
        # Now restored_struct is a new gemmi.Structure with matching geometry
    """

    def serialize_structure(self, structure: gemmi.Structure) -> dict:
        """
        Convert a Gemmi Structure into a Python dict of primitive types.
        
        The returned dictionary can be JSON/pickle‐serialized, 
        or used in your logging system for debugging/state capture.
        """
        data = {}
        
        # 1. Cell parameters
        cell = structure.cell
        data["cell"] = {
            "a": cell.a,
            "b": cell.b,
            "c": cell.c,
            "alpha": cell.alpha,
            "beta": cell.beta,
            "gamma": cell.gamma
        }
        
        # 2. Space group
        data["space_group"] = structure.spacegroup_hm  # e.g. 'P 1 21 1'
        
        # 3. Models, Chains, Residues, Atoms
        models_list = []
        for model in structure:
            model_dict = {
                "name": model.name,
                "chains": []
            }
            
            for chain in model:
                chain_dict = {
                    "name": chain.name,
                    "residues": []
                }
                
                for res in chain:
                    residue_dict = {
                        "name": res.name,  # e.g. 'ALA'
                        # optional: insertion/resid parsing
                        "seqid_num": res.seqid.num,
                        "seqid_icode": res.seqid.icode,
                        "atoms": []
                    }
                    
                    for atom in res:
                        atom_dict = {
                            "name": atom.name,
                            # position
                            "x": atom.pos.x,
                            "y": atom.pos.y,
                            "z": atom.pos.z,
                            # B‐factor, occupancy, etc.
                            "occ": atom.occ,
                            "bfactor": atom.b_iso,
                            "element": atom.element.name  # e.g. 'C'
                        }
                        residue_dict["atoms"].append(atom_dict)
                    
                    chain_dict["residues"].append(residue_dict)
                
                model_dict["chains"].append(chain_dict)
            
            models_list.append(model_dict)
        
        data["models"] = models_list
        
        return data

    def deserialize_structure(self, data: dict) -> gemmi.Structure:
        """
        Rebuild a gemmi.Structure from the dictionary produced by serialize_structure().
        """
        structure = gemmi.Structure()
        
        # 1. Cell
        cell_data = data.get("cell", {})
        if cell_data:
            structure.cell = gemmi.UnitCell(
                cell_data.get("a", 0.0),
                cell_data.get("b", 0.0),
                cell_data.get("c", 0.0),
                cell_data.get("alpha", 90.0),
                cell_data.get("beta", 90.0),
                cell_data.get("gamma", 90.0)
            )
        
        # 2. Space group
        space_group = data.get("space_group", None)
        if space_group:
            structure.spacegroup_hm = space_group
        
        # 3. Models, Chains, etc.
        models_list = data.get("models", [])
        for model_dict in models_list:
            model = gemmi.Model(model_dict["name"])
            
            for chain_dict in model_dict["chains"]:
                chain = gemmi.Chain(chain_dict["name"])
                
                for res_dict in chain_dict["residues"]:
                    res = gemmi.Residue()
                    res.name = res_dict["name"]
                    
                    # optional: restore seqid
                    seqid_num = res_dict.get("seqid_num", 0)
                    seqid_icode = res_dict.get("seqid_icode", "")
                    seqid = gemmi.SeqId(seqid_num, seqid_icode)
                    res.seqid = seqid
                    
                    # atoms
                    for atom_dict in res_dict["atoms"]:
                        atom = gemmi.Atom()
                        atom.name = atom_dict["name"]
                        atom.pos = gemmi.Position(
                            atom_dict["x"],
                            atom_dict["y"],
                            atom_dict["z"]
                        )
                        atom.occ = atom_dict.get("occ", 1.0)
                        atom.b_iso = atom_dict.get("bfactor", 0.0)
                        elem_name = atom_dict.get("element", "X")
                        atom.element = gemmi.Element(elem_name)
                        
                        res.add_atom(atom)
                    
                    chain.add_residue(res)
                
                model.add_chain(chain)
            
            structure.add_model(model)
        
        return structure


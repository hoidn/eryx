# test_gemmi_serializer.py

import unittest
import gemmi
from eryx.autotest.gemmi_serializer import GemmiSerializer

class TestGemmiSerializer(unittest.TestCase):

    def setUp(self):
        # Build a small gemmi.Structure from scratch
        self.structure = gemmi.Structure()
        
        # Setup cell
        self.structure.cell = gemmi.UnitCell(55.0, 66.0, 77.0, 90.0, 100.0, 120.0)
        self.structure.spacegroup_hm = 'P 2 2 2'
        
        # Create one model
        model = gemmi.Model("test_model")
        
        # One chain
        chain = gemmi.Chain("A")
        
        # One residue
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(10, " ")


        # One atom
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.pos = gemmi.Position(1.234, 2.345, 3.456)
        atom.occ = 0.9
        atom.b_iso = 20.0
        atom.element = gemmi.Element("C")
        
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
        self.structure.add_model(model)

    def test_serialize_deserialize(self):
        serializer = GemmiSerializer()
        
        # Serialize
        serialized_dict = serializer.serialize_structure(self.structure)
        
        # Basic checks
        self.assertIn("cell", serialized_dict)
        self.assertIn("models", serialized_dict)
        
        # Check cell contents
        self.assertAlmostEqual(serialized_dict["cell"]["a"], 55.0)
        self.assertAlmostEqual(serialized_dict["cell"]["beta"], 100.0)
        
        # Check model/chain residue
        self.assertEqual(len(serialized_dict["models"]), 1)
        model_dict = serialized_dict["models"][0]
        self.assertEqual(model_dict["name"], "test_model")
        
        self.assertEqual(len(model_dict["chains"]), 1)
        chain_dict = model_dict["chains"][0]
        self.assertEqual(chain_dict["name"], "A")
        
        self.assertEqual(len(chain_dict["residues"]), 1)
        residue_dict = chain_dict["residues"][0]
        self.assertEqual(residue_dict["name"], "ALA")
        self.assertEqual(residue_dict["seqid_num"], 10)
        
        self.assertEqual(len(residue_dict["atoms"]), 1)
        atom_dict = residue_dict["atoms"][0]
        self.assertEqual(atom_dict["name"], "CA")
        self.assertAlmostEqual(atom_dict["x"], 1.234, places=3)
        
        # Now deserialize
        restored_structure = serializer.deserialize_structure(serialized_dict)
        
        # Compare key aspects of the round-trip
        self.assertAlmostEqual(restored_structure.cell.a, 55.0)
        self.assertAlmostEqual(restored_structure.cell.beta, 100.0)
        self.assertEqual(restored_structure.spacegroup_hm, 'P 2 2 2')
        
        # check model
        self.assertEqual(len(restored_structure), 1)  # 1 model
        model_rest = restored_structure[0]
        self.assertEqual(model_rest.name, "test_model")
        
        # chain
        self.assertEqual(len(model_rest), 1)
        chain_rest = model_rest[0]
        self.assertEqual(chain_rest.name, "A")
        
        # residue
        self.assertEqual(len(chain_rest), 1)
        res_rest = chain_rest[0]
        self.assertEqual(res_rest.name, "ALA")
        self.assertEqual(res_rest.seqid.num, 10)
        
        # atom
        self.assertEqual(len(res_rest.atoms), 1)
        atom_rest = res_rest.atoms[0]
        self.assertEqual(atom_rest.name, "CA")
        self.assertAlmostEqual(atom_rest.pos.x, 1.234, places=3)
        self.assertAlmostEqual(atom_rest.b_iso, 20.0)
        self.assertEqual(atom_rest.element.name, "C")


if __name__ == '__main__':
    unittest.main()


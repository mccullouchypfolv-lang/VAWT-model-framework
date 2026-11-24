import copy,shutil,unittest,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
from lf_operator import hf_dataset
from lf_operator.data import digest


class HFDatasetTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]/'tmp'/('hf100-test-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        for name in ['old/S01','new/S02','new/_incomplete','cache/cases']:(self.root/name).mkdir(parents=True)
        self.cfg=dict(hf_roots=['old','new'],expected_hf_cases=2)

    def cleanup(self):
        assert self.root.resolve().is_relative_to((Path(__file__).resolve().parents[1]/'tmp').resolve())
        shutil.rmtree(self.root)

    def test_collect_multiple_roots_is_sorted_and_rejects_duplicates_or_counts(self):
        folders=hf_dataset.collect_hf_folders(self.cfg,self.root)
        self.assertEqual([p.name for p in folders],['S01','S02'])
        with self.assertRaises(ValueError):hf_dataset.collect_hf_folders(dict(self.cfg,expected_hf_cases=3),self.root)
        (self.root/'new/S01').mkdir()
        with self.assertRaisesRegex(ValueError,'duplicate'):hf_dataset.collect_hf_folders(self.cfg,self.root)

    def test_build_matches_cases_and_preserves_labels_and_source_locations(self):
        theta=(np.arange(160)+.5)*2*np.pi/160;records=[];samples={}
        for i,name in enumerate(['S01','S02']):
            file=f'cases/{name}.npz';a=np.array([10.,2.+i]);reference=np.array([2.+i,10.,3.,.18])
            np.savez(self.root/'cache'/file,a=a,reference_conditions=reference,theta=theta)
            records.append(dict(case_id=name,geometry_id=f'G{i}',file=file,cache_hash=digest(self.root/'cache'/file)))
            samples[name]=dict(case_id=name,conditions=a,reference_conditions=reference,theta=theta,y=np.ones((160,3))*i,metadata=dict(source_hashes={'test':'hash'}))
        manifest=dict(records=records,config={})
        folders=hf_dataset.collect_hf_folders(self.cfg,self.root)
        with patch('lf_operator.blade_data.read_hf_case',side_effect=lambda path,config:copy.deepcopy(samples[path.name])):
            arrays,provenance=hf_dataset.build_hf_arrays(folders,self.root/'cache',manifest)
            np.testing.assert_array_equal(arrays['target'],np.stack([samples[n]['y'] for n in ['S01','S02']]))
            self.assertEqual(provenance['S02']['source_directory'],str((self.root/'new/S02').resolve()))
            samples['S02']['conditions']=np.array([10.,9.])
            with self.assertRaisesRegex(ValueError,'condition'):hf_dataset.build_hf_arrays(folders,self.root/'cache',manifest)
        with self.assertRaisesRegex(ValueError,'LF'):hf_dataset.build_hf_arrays(folders,self.root/'cache',dict(manifest,records=records[:1]))


if __name__=='__main__':unittest.main()

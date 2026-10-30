import io
import json
import unittest
import zipfile
from reevaluate_entity_rate import parse_outcome, summarize


class EvaluationTests(unittest.TestCase):
    def test_first_pass_and_repair_are_separate(self):
        records=[{'id':'one','first_pass_valid':True,'valid_solid':True},
                 {'id':'two','first_pass_valid':False,'valid_solid':True},
                 {'id':'three','first_pass_valid':False,'valid_solid':False}]
        summary=summarize(records,['one','two','three'])
        self.assertEqual(summary['first_pass_valid_count'],1)
        self.assertEqual(summary['repair_added_valid_count'],1)
        self.assertEqual(summary['final_valid_rate'],2/3)

    def test_partial_run_has_no_final_rate(self):
        summary=summarize([{'id':'one','first_pass_valid':True,'valid_solid':True}],['one','two'])
        self.assertIsNone(summary['final_valid_rate'])
        self.assertEqual(summary['pending_ids'],['two'])

    def test_success_requires_valid_geometry_and_trace(self):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            z.writestr('model.step',b'ISO-10303-21;fixture')
            z.writestr('model.py','fixture-not-executed')
            z.writestr('generation.json',json.dumps({'geometry':{'valid':True},'repair_count':1}))
        record,_,_=parse_outcome(200,output.getvalue(),{'status':'completed','attempts':[{'status':'failed'},{'status':'passed'}]})
        self.assertFalse(record['first_pass_valid'])
        self.assertTrue(record['valid_solid'])
        with self.assertRaises(AssertionError):parse_outcome(200,output.getvalue(),{'status':'failed'})

    def test_clarification_counts_as_no_solid(self):
        record,step,_=parse_outcome(422,json.dumps({'detail':{'code':'needs_clarification'}}).encode(),{})
        self.assertFalse(record['valid_solid'])
        self.assertIsNone(step)


if __name__=='__main__':unittest.main()

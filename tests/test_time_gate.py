from dataclasses import replace
import unittest

from specnet.contracts import Policy, Subquery, TaskContract
from specnet.core import Coordinator
from specnet.profiles import Prediction
from tests.test_core import Clock, candidate


class TimeGateTests(unittest.TestCase):
    def setup_engine(self,enabled=True,**limits):
        self.clock=Clock()
        self.engine=Coordinator(clock=self.clock,snapshot_max_age=1,time_risk_enabled=enabled)
        contract=TaskContract('t','question',(Subquery('q','question'),),deadline_seconds=10,
            final_reserve_seconds=2,observation_seconds=12,optional_limit=3,optional_payload_budget=12,
            idle_wait_seconds=.3)
        contract=replace(contract,**limits)
        self.engine.register(contract)
        self.engine.set_candidates('t','q',[candidate(i) for i in range(4)])
        job=self.engine.admit_base('t','q')
        self.engine.start_fetch(job.id);self.engine.fetch_done(job.id);self.engine.start_extract(job.id)
        self.engine.confirmed_terminal(job.id,'success','base')
        self.engine.open_optional('t')

    def commit(self,prediction):
        c=self.engine.next_candidate('t');s=self.engine.snapshot(telemetry_at=self.clock(),domain='fixture')
        return self.engine.try_commit('t',c,s,prediction,Policy.PROGRESSIVE)

    def test_only_time_inequality_is_disabled(self):
        self.setup_engine(True)
        self.assertEqual(self.commit(Prediction(100,4)).reason,'time_risk')
        self.setup_engine(False)
        self.assertEqual(self.commit(Prediction(100,4)).action,'ADMIT_NEXT')

    def test_unknown_and_idle_wait_remain_active(self):
        self.setup_engine(False)
        self.assertEqual(self.commit(Prediction(None,None,'telemetry_stale',True)).action,'DEFER')
        self.clock.advance(.31)
        self.assertEqual(self.commit(Prediction(None,None,'telemetry_stale',True)).reason,'idle_wait_exhausted')

    def test_budget_concurrency_and_cutoff_remain_active(self):
        self.setup_engine(False,optional_payload_budget=3)
        self.assertEqual(self.commit(Prediction(100,4)).reason,'reservation_does_not_fit')
        self.setup_engine(False,optional_concurrency=1)
        self.assertEqual(self.commit(Prediction(100,4)).action,'ADMIT_NEXT')
        self.assertEqual(self.commit(Prediction(100,4)).reason,'concurrency')
        self.setup_engine(False)
        self.clock.advance(8)
        self.assertEqual(self.commit(Prediction(100,4)).reason,'cutoff')


if __name__=='__main__':unittest.main()

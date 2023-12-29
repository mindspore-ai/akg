"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
from tests.common.base import TestBase
from tests.common.test_run.ascend.col2im_run import col2im_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_tensor_col2im_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path, 0)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype,output_h_w,polyhedral,attr
            ("mansch_col2im", col2im_run, ((1, 1, 3, 3, 8, 8, 16), (3, 3), (2, 2), (2, 1, 2, 1), (14, 14), "float32")),
            ("mansch_col2im", col2im_run, ((1, 1, 2, 2, 8, 8, 16), (2, 2), (2, 2), (0, 0, 0, 0), (16, 16), "float32")),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()

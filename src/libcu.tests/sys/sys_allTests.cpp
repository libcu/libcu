#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t sys_stat_test1();
cudaError_t sys_time_test1();

namespace libcutests {
	// sys_stat
	__BEGIN_TEST(sys_stat);
	[TestMethod, TestCategory("core")] void sys_stat_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::sys_stat_test1()))); }
	__END_TEST;

	// sys_time
	__BEGIN_TEST(sys_time);
	[TestMethod, TestCategory("core")] void sys_time_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::sys_time_test1()))); }
	__END_TEST;
}

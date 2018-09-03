#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t ext_hash_test1();
cudaError_t ext_memfile_test1();

namespace libcutests {
	// ext_hash
	__BEGIN_TEST(ext_hash);
	[TestMethod, TestCategory("core")] void ext_hash_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::ext_hash_test1()))); }
	__END_TEST;

	// ext_memfile
	__BEGIN_TEST(ext_memfile);
	[TestMethod, TestCategory("core")] void ext_memfile_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::ext_memfile_test1()))); }
	__END_TEST;
}

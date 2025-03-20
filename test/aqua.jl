using Aqua
using FittedItemBanks

Aqua.test_all(FittedItemBanks, ambiguities = false, deps_compat = false)
Aqua.test_ambiguities([FittedItemBanks])
Aqua.test_deps_compat(FittedItemBanks; check_extras = false)

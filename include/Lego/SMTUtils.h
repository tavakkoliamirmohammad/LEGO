#ifndef LEGO_SMTUTILS_H
#define LEGO_SMTUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <string>
#include <optional>

namespace mlir {
namespace lego {

struct SMTBuilder {
  OpBuilder &builder;
  DenseMap<Value, Value> valMap;
  AsmState &state;
  unsigned &nextId;

  SMTBuilder(OpBuilder &b, AsmState &s, unsigned &nextId)
    : builder(b), state(s), nextId(nextId) {}

  std::string getSSAName(Value v);
  Value getOrCreate(Value v);
  Value buildTruncDiv(Value a, Value b, Location loc);
  void buildRegion(Region &region, ValueRange args, SmallVectorImpl<Value> &results);
};

// Result of an SMT query
struct SMTResult {
  bool isSat;
  bool isUnsat;
  bool isUnknown;
  llvm::StringMap<int64_t> model; // Counter-example model if SAT
  std::string rawOutput;

  SMTResult() : isSat(false), isUnsat(false), isUnknown(false) {}
};

// Abstracted context for configuring and extracting SMT formulas
struct SMTSolverContext {
  OwningOpRef<ModuleOp> smtModule;
  std::unique_ptr<OpBuilder> b;
  std::unique_ptr<SMTBuilder> builder;
  Location loc;

  SMTSolverContext(Location loc, AsmState &state, unsigned &nextId);
  ~SMTSolverContext() {}

  // Finalizes the check commands, exports the SMT-LIB and tests solving.
  // timeoutMs: Z3 timeout in milliseconds (0 = no timeout, default = 30000).
  SMTResult checkSatisfiability(const SmallVector<std::string> &varNamesToExtract,
                                unsigned timeoutMs = 30000);
};

// Run Z3 and return detailed result. timeoutMs=0 means no timeout.
SMTResult runZ3WithModel(const std::string &smtLib, unsigned timeoutMs = 30000);

// Legacy interface (returns true if SAT)
bool runZ3(const std::string &smtLib);

// Helper to generate SMT-LIB get-value commands for a list of variables
std::string generateGetValueCommands(const SmallVector<std::string> &varNames);

/// Compute per-thread flat addresses for a warp of threads accessing
/// a layout via an ApplyOp.  Builds SMT expressions for each thread's
/// flat index by substituting (baseThread + t) for the lego.thread_id
/// argument and evaluating the layout body.
///
/// Supports all layout types: GenP (via region evaluation), RegP,
/// OrderBy, GroupBy, and TileBy (via materializing arith ops and
/// encoding them into SMT).
///
/// Returns success() and fills \p addresses (one per thread) and
/// \p baseThread.  Returns failure() if the layout cannot be lowered.
LogicalResult computeWarpAddresses(
    Operation *apply, Value layout, ValueRange indices,
    SMTSolverContext &smtCtx, AsmState &state, unsigned &nextId,
    int warpSize, Value &baseThread, SmallVectorImpl<Value> &addresses);

} // namespace lego
} // namespace mlir

#endif // LEGO_SMTUTILS_H

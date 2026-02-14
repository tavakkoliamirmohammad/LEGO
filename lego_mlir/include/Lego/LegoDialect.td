#ifndef LEGO_DIALECT
#define LEGO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LEGO Dialect
//===----------------------------------------------------------------------===//

def Lego_Dialect : Dialect {
  let name = "lego";
  let summary = "A dialect for LEGO layout optimizations";
  let description = [{
    The LEGO dialect provides operations to define and apply data layout transformations
    using the LEGO (Layouts for Efficient GPU Optimization) framework.
  }];
  let cppNamespace = "::mlir::lego";
  
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// LEGO Types
//===----------------------------------------------------------------------===//

def Lego_LayoutType : TypeDef<Lego_Dialect, "Layout"> {
    let mnemonic = "layout";
    let summary = "A LEGO layout definition";
    let description = "Represents a layout object (e.g., Row-Major, Tiled, etc.)";
}

#endif // LEGO_DIALECT

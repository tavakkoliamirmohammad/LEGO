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

def Lego_ViewType : TypeDef<Lego_Dialect, "View"> {
    let mnemonic = "view";
    let summary = "A structured view over a memory buffer";
    let description = [{
        Represents a logical view over a physical memory buffer, defined by a Lego layout.
        The view abstraction allows multidimensional access to flat or complex buffers
        abstracting away the index calculation logic.
    }];
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = "`<` $elementType `>`";
}

#endif // LEGO_DIALECT

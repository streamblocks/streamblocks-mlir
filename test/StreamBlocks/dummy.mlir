// RUN: streamblocks-opt %s | streamblocks-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = dwf.foo %{{.*}} : i32
        %res = dwf.foo %0 : i32
        return
    }
}

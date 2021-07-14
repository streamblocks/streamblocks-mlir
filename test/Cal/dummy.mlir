// RUN: cal-opt %s | cal-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = cal.foo %{{.*}} : i32
        %res = cal.foo %0 : i32
        return
    }
}

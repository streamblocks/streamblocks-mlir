cal.namespace {
     cal.actor @Println () -> () {
        cal.process {
            %0 = constant 1 : i32
            cal.print %0 : i32
        }
    }
}

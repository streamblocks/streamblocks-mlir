cal.namespace {
     cal.actor @Println () -> () {
        cal.process {
            %0 = cal.constant 42 : i32
            cal.println %0 : i32
        } {repeat = false}
    }
} {qid = "test"}

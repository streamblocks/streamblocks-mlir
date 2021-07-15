module {
     cal.actor @Process() -> () {
        cal.process attributes {repeat = false} {
            %0 = cal.constant 1 : %i32
            cal.println %0 : i32
        }
    }
}

am.program @Println () -> () {

    am.transition @0 () -> () {
        %0 = constant 42 : i32
        am.println %0 : i32
    }

    am.controller {

        am.state @S0 {
            am.exec @0 -> @S1
        }

        am.state @S1 {
            am.exit
        }
    }
}
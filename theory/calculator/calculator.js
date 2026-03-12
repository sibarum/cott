
;(() => {

    const buttonCodes = {};

    buttonCodes["&l"] = ({cursor}) => {
        const newTarget = cursor.previousElementSibling;
        if (newTarget) {
            cursor.remove();
            newTarget.before(cursor);
        }
    };

    buttonCodes["&r"] = ({cursor}) => {
        const newTarget = cursor.nextSibling;
        if (newTarget) {
            cursor.remove();
            newTarget.after(cursor);
        }
    };

    buttonCodes["&del"] = ({cursor, display}) => {
        const newTarget = cursor.previousElementSibling;
        if (newTarget) {
            newTarget.remove();
        } else {
            if (cursor.parentNode.id !== display.id) {
                const newSibling = cursor.parentNode.previousElementSibling || cursor.parentNode.nextSibling;
                cursor.parentNode.remove();
                newSibling.after(cursor);
            }
        }
    };

    buttonCodes["&clr"] = ({cursor}) => {
        display.replaceChildren();
        display.append(cursor);
    };

    buttonCodes["&sqrt"] = ({cursor}) => {
    };

    buttonCodes["&exp"] = ({cursor}) => {
        const superscript = document.createElement("sup");
        superscript.textContent = " ";
        cursor.parentNode.insertBefore(superscript, cursor);
        cursor.remove();
        superscript.append(cursor);
    };

    buttonCodes["&log"] = ({cursor}) => {
    };



    function calculatorInput(value) {
        const display = document.getElementById("primary-display");
        const cursor = document.getElementById("calculator-cursor");

        if (value in buttonCodes) {
            buttonCodes[value]({cursor, display});
        } else {
            const newElement = document.createElement("span")
            if (value === 'w') {
                newElement.classList.add("symbol");
                newElement.innerHTML = "&omega;";
            } else {
                newElement.classList.add("number");
                newElement.textContent = value;
            }
            cursor.parentNode.insertBefore(newElement, cursor);
        }

    }

    function initializeDisplay() {
        const display = document.getElementById("primary-display");
        display.addEventListener("click", () =>
        {

        });
    }

    function initializeCalculator() {
        initializeDisplay();
        let numpad = document.getElementById("calculator-numpad");
        let buttons = numpad.getElementsByClassName("btn");
        numpad = document.getElementById("calculator-actionbar");
        buttons = [...buttons, ...numpad.getElementsByClassName("btn")];
        for (let button of buttons) {
            if (button.dataset && button.dataset.value) {
                button.addEventListener("click", () => {
                    calculatorInput(button.dataset.value);
                });
            }
        }
    }

    window.addEventListener("load", () => {
        initializeCalculator();
    });
})();
void setup() {
  pinMode(13, OUTPUT);    // sets the digital pin 13 as output
  digitalWrite(13, LOW);  // sets the digital pin 13 off
  Serial.begin(115200);
}

void loop() {
  // read serial input
  int inputchar;
  inputchar = Serial.read();

  if(inputchar!=-1){
    switch(inputchar){
      case 's':
        digitalWrite(13,HIGH);
        break;

      case 'q':
        digitalWrite(13,LOW);
        break;
    }
  }
  else{
  }
}

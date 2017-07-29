
import java.util.*;
public class JSONLibrary {
    public static void main(String[] args) {
        int valid=1;
        int c=1;
        int count=0,colon_c=0;
        String temp;
        String open_b="{";
        String close_b="}";
        String open_s="[";
        String close_s="]";
        String double_q="\"";
        String colon=":";
       String str="{\n" +
"    \"name\":\"John\",\n" +
"    \"age\":30,\n" +
"    \"cars\": [\n" +
"        { \"name\":\"Ford\", \"models\":[ \"Fiesta\", \"Focus\", \"Mustang\" ] },\n" +
"        { \"name\":\"BMW\", \"models\":[ \"320\", \"X3\", \"X5\" ] },\n" +
"        { \"name\":\"Fiat\", \"models\":[ \"500\", \"Panda\" ] }\n" +
"    ]\n" +
" }"; 
        System.out.println(str);
        Stack<String> st=new Stack<>();
         for (int i = 0; i < str.length(); i++) {
        if (str.charAt(i) == '{') {
            
            temp=""+str.charAt(i);
            st.push(temp);
        }
        else if(st.empty() && str.charAt(i)!='{'){
            
            valid=0;
            break;
        }
        else if(str.charAt(i)=='"' && open_b.equals(st.peek())){
            temp=""+str.charAt(i);  
            st.push(temp);
        }
        else if(str.charAt(i)=='"' && double_q.equals(st.peek())){   
            st.pop();
            c=1;
            count=1;
            colon_c++;
        }
        else if(str.charAt(i)==':' && c==1){
            c=0;
            count++;
        }
        else if(str.charAt(i)=='"' && (count==2) && (c==0)){
            temp=""+str.charAt(i);  
            st.push(temp);
        }
        else if(str.charAt(i)==',' && (count==2) || (count==1)){
            count=0;
            c=0;
        }
        else if((str.charAt(i)=='[')){
            temp=""+str.charAt(i);  
            st.push(temp);
        }
        else if(str.charAt(i)=='{' && open_s.equals(st.peek())){
            temp=""+str.charAt(i);  
            st.push(temp);
        }
        else if(str.charAt(i)=='"' && count==0 && c==0){
            temp=""+str.charAt(i);  
            st.push(temp);
        }
        else if(str.charAt(i)=='}' && open_b.equals(st.peek())){
            st.pop();
        }
        else if(str.charAt(i)==']' && open_s.equals(st.peek())){
            st.pop();
        }
        else if((str.charAt(i)>=48 && str.charAt(i)<=57) || (str.charAt(i)>=65 && str.charAt(i)<=92) || (str.charAt(i)>=97 && str.charAt(i)<=122) || 
                (str.charAt(i)=='+') || (str.charAt(i)=='\n') || (str.charAt(i)=='\\') || (str.charAt(i)==' ') || (str.charAt(i)==',')){
            valid=1;
        }
        else{
            valid=0;
            break;
        }
    }   
         printStack(st);
         if(st.empty() && valid!=0){
             System.out.println("Valid JSON");
         }
         else {
             System.out.println("JSON IS INVALID");
         }
    }
    
    private static void printStack(Stack<String> s) {
        if(s.empty()){
            System.out.println("Stack is empty");
        }
        else{
        System.out.printf("Bottom:%s:TOP\n",s);
        }
    }
    
}

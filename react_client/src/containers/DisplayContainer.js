import React from 'react';
import request from 'sync-request';


var buttonStyle = {
    margin: '10px 10px 10px 0'
};

var state = {
    labels:[],
    datasets: [],
}

class DispContainer extends React.Component{

    constructor(props){
        super();

    }
    getTangleData(){
        var res = request('GET','http://localhost:5000/getReportData');
        console.log('Got data!')
        return 0;
    }

    shutDownServer(){
        var res = request('GET','http://localhost:5000/shutdown');
        console.log('Shut down server!')
    }


    render(){

        return(
            <div>
                <row style={{
                    display:'flex',
                    justifyContent:"center",
                    alignItems:'center'
                }}>
                    <button
                        className="btn btn-default"
                        style={buttonStyle}
                        onClick={this.getTangleData}>Save Data</button>
                    <button
                    className="btn btn-default"
                    style={buttonStyle}
                    onClick={this.shutDownServer}>Shutdown and Save Plots</button>
                </row>

            </div>
        );

    }
}

export default DispContainer;

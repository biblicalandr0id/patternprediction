import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ParameterAnalysis = () => {
  // Simulated completion accuracy data for different parameter sets
  const [data] = useState([
    {
      removalRate: 30,
      default: 92,
      fastAdaptation: 88,
      highSensitivity: 95,
    },
    {
      removalRate: 50,
      default: 85,
      fastAdaptation: 82,
      highSensitivity: 89,
    },
    {
      removalRate: 70,
      default: 75,
      fastAdaptation: 70,
      highSensitivity: 80,
    },
    {
      removalRate: 90,
      default: 60,
      fastAdaptation: 55,
      highSensitivity: 65,
    },
  ]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Pattern Completion Accuracy Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="removalRate" 
                label={{ value: 'Pattern Removal Rate (%)', position: 'bottom' }} 
              />
              <YAxis 
                label={{ value: 'Completion Accuracy (%)', angle: -90, position: 'left' }}
                domain={[50, 100]} 
              />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="default" 
                stroke="#8884d8" 
                name="Default Parameters"
              />
              <Line 
                type="monotone" 
                dataKey="fastAdaptation" 
                stroke="#82ca9d" 
                name="Fast Adaptation"
              />
              <Line 
                type="monotone" 
                dataKey="highSensitivity" 
                stroke="#ffc658" 
                name="High Sensitivity"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 space-y-2">
          <h3 className="text-lg font-semibold">Parameter Configurations:</h3>
          <div className="space-y-1">
            <p><strong>Default:</strong> Balanced parameters (threshold: 0.5, adaptation: 0.1)</p>
            <p><strong>Fast Adaptation:</strong> Quicker response (threshold: 0.4, adaptation: 0.2)</p>
            <p><strong>High Sensitivity:</strong> Lower threshold (threshold: 0.3, adaptation: 0.08)</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ParameterAnalysis;